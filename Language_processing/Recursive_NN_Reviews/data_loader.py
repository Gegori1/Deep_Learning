from torch.utils.data import DataLoader
from data_set import DatasetReviews
from sklearn.model_selection import train_test_split

def dataloader_reviews_train_eval(sequences, vocab_size: int, batch_size: int, num_workers: int, eval_size: float= 0.3, shuffle: bool=True, random_seed: bool=42, device: object=...):
    
    X, Y = sequences
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=eval_size, random_state=random_seed)
    
    train_dataset = DatasetReviews(X_train, y_train, vocab_size)
    val_dataset = DatasetReviews(X_val, y_val, vocab_size)
    
    if 'cuda' in device.type:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True, pin_memory_device=device.type)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True, pin_memory_device=device.type)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    
    return train_loader, val_loader

def dataloader_reviews_test(X, Y, vocab_size: int, batch_size: int, num_workers: int, shuffle: bool=True, device='gpu'):
        
        test_dataset = DatasetReviews(X, Y, vocab_size)
        
        if 'cuda' in device.type:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True, pin_memory_device=device.type)
        else:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        
        return test_loader