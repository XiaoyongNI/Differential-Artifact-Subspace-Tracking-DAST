"""Clean-only training, validation selection, and frozen matched inference."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from .common import derived_seed, seed_everything, write_csv, write_json
from .models import build_model


def fit_normalization(clean_train, offset, epsilon):
    # Stable chunked parallel moments; never use validation or test data.
    count = 0
    mean = np.zeros(clean_train.shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    for window in clean_train:
        block = np.asarray(window[:, offset:], dtype=np.float64)
        n = block.shape[1]
        bmean = block.mean(axis=1)
        delta = bmean-mean
        m2 += ((block-bmean[:, None])**2).sum(axis=1) + delta**2 * count*n/(count+n)
        mean += delta*n/(count+n)
        count += n
    scale = np.maximum(np.sqrt(m2/count), epsilon)
    return mean.astype(np.float32), scale.astype(np.float32)


class SignalWindows(Dataset):
    def __init__(self, x, y, mean, scale, offset=0):
        self.x, self.y, self.offset = x, y, offset
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.mean.shape != (x.shape[1],) or self.scale.shape != self.mean.shape or np.any(self.scale <= 0):
            raise ValueError("Normalization must contain one positive scale and mean per channel")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        signal = np.array(self.x[index, :, self.offset:], dtype=np.float32, copy=True)
        signal = (signal-self.mean[:, None])/self.scale[:, None]
        if not np.isfinite(signal).all():
            raise ValueError(f'Nonfinite decoder input at window {index}')
        return torch.from_numpy(signal), int(self.y[index])


def selected_device(config):
    device = config['training']['device']
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)


def _train(name, train_x, train_y, val_x, val_y, normalization, config, output_dir, patient):
    options = config['training']
    device = selected_device(config)
    offset = config['evaluation_offset_samples']
    mean, scale = normalization
    train = SignalWindows(train_x, train_y, mean, scale, offset)
    validation = SignalWindows(val_x, val_y, mean, scale, offset)
    if set(np.unique(train_y)) != {0,1}:
        raise ValueError('Clean training split must contain both classes')
    output_dir = Path(output_dir)/name
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(train_y, minlength=2)
    weights = (len(train_y)/(2*counts)).astype(np.float32) if options['class_weighted_loss'] else np.ones(2, np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), reduction='none')
    best_overall, best_path, selection = np.inf, None, None
    history = []
    for trial, lr in enumerate(options['learning_rates']):
        seed = derived_seed(config['seed'], patient, name, 'training', trial)
        seed_everything(seed)
        model = build_model(name, train_x.shape[1], train_x.shape[-1]-offset, config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=options['weight_decay'])
        train_loader = DataLoader(train, batch_size=options['batch_size'], shuffle=True,
            generator=torch.Generator().manual_seed(seed), num_workers=0)
        val_loader = DataLoader(validation, batch_size=options['batch_size'], shuffle=False, num_workers=0)
        best_loss, patience_reference, stale = np.inf, np.inf, 0
        checkpoint = output_dir/f'candidate_{trial}_best.pt'
        for epoch in range(options['epochs']):
            model.train()
            total, n = 0.0, 0
            for signals, labels in train_loader:
                signals, labels = signals.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(signals), labels).mean()
                if not torch.isfinite(loss):
                    raise ValueError(f'{patient}/{name}: nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip'])
                optimizer.step()
                total += loss.item()*len(labels)
                n += len(labels)
            model.eval()
            val_total = 0.0
            with torch.inference_mode():
                for signals, labels in val_loader:
                    loss = criterion(model(signals.to(device)), labels.to(device)).sum()
                    val_total += loss.item()
            val_loss = val_total/len(validation)
            if not np.isfinite(val_loss):
                raise ValueError(f'{patient}/{name}: nonfinite validation loss')
            history.append(dict(candidate=trial, learning_rate=lr, seed=seed, epoch=epoch+1,
                                train_loss=total/n, validation_loss=val_loss))
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    decoder=name, channels=train_x.shape[1], samples=train_x.shape[-1]-offset,
                    mean=torch.from_numpy(mean), scale=torch.from_numpy(scale),
                    epoch=epoch+1, validation_loss=val_loss, learning_rate=lr, seed=seed,
                    config=config, class_weights=weights.tolist()), checkpoint)
            if val_loss < patience_reference-options['min_delta']:
                patience_reference, stale = val_loss, 0
            else:
                stale += 1
            if stale >= options['patience']:
                break
        if best_loss < best_overall:
            best_overall, best_path = best_loss, checkpoint
            selection = dict(candidate=trial, learning_rate=lr, validation_loss=best_loss, seed=seed)
        print(f'{patient} {name} candidate={trial} best clean validation loss={best_loss:.5f}', flush=True)
    import shutil
    shutil.copyfile(best_path, output_dir/'best.pt')
    saved = torch.load(output_dir/'best.pt', map_location='cpu', weights_only=True)
    model = build_model(name, train_x.shape[1], train_x.shape[-1]-offset, config)
    model.load_state_dict(saved['model'])
    model.to(device).eval().requires_grad_(False)
    write_csv(output_dir/'training_history.csv', history)
    write_json(output_dir/'selection.json', dict(**selection, epoch=saved['epoch'],
        criterion='Minimum clean validation weighted cross-entropy; fixed argmax decision rule',
        validation_classes=np.unique(val_y).tolist(), frozen=True))
    return model


def train_eegnet(*args, **kwargs):
    return _train('EEGNet', *args, **kwargs)


def train_gru(*args, **kwargs):
    return _train('GRU', *args, **kwargs)


def train_chrononet(*args, **kwargs):
    return _train('ChronoNet', *args, **kwargs)


def evaluate_decoder(model, x, labels, normalization, config, offset=0):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Evaluation requires a frozen decoder in eval mode')
    mean, scale = normalization
    loader = DataLoader(SignalWindows(x, labels, mean, scale, offset),
                        batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
    device = next(model.parameters()).device
    probabilities = []
    with torch.inference_mode():
        for signals, _ in loader:
            logits = model(signals.to(device))
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite prediction')
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    predicted = probabilities.argmax(axis=1)
    # Kappa is undefined if both truth and predictions contain one shared class.
    degenerate = len(np.unique(np.concatenate([labels, predicted]))) == 1
    metrics = dict(accuracy=float(accuracy_score(labels, predicted)),
                   weighted_f1=float(f1_score(labels, predicted, average='weighted', zero_division=0)),
                   kappa=float('nan') if degenerate else float(cohen_kappa_score(labels, predicted)))
    return metrics, probabilities
