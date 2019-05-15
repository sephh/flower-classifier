from time import time

from classifier import get_pretrained_model, save_model
from train_func import load_data, train, get_train_args

def main():
    # Command Line Args
    in_args = get_train_args()
    
    device = 'cuda' if in_args.gpu else 'cpu'
    
    # Start Time
    start_time = time()
    
    # Load train, test and validation datasets
    dataloaders, image_datasets, data_transforms = load_data(in_args.data_dir)
    
    # Get the model
    model, hidden_units, dropout = get_pretrained_model(in_args.arch, in_args.hidden_units, in_args.dropout)
    
    # Training
    train(
        model=model, 
        dataloaders=dataloaders, 
        device=device, 
        lr=in_args.learning_rate, 
        epochs=in_args.epochs
    )
    
    # Save checkpoint
    save_model(
        model=model,
        arch=in_args.arch, 
        hidden_units=hidden_units, 
        dropout=dropout, 
        class_to_idx=image_datasets['train'].class_to_idx, 
        epochs=in_args.epochs, 
        path=in_args.save_dir
    )
    
    # End Time
    end_time = time()
    
    # Caluculate elapsed time
    tot_time = end_time - start_time
    elapsed_hours = int((tot_time/3600))
    elapsed_minutes = int((tot_time%3600)/60)
    elapsed_seconds = int((tot_time%3600)%60)
    
    print("\n** Total Elapsed Runtime:", f'{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}')
    
# Call to main function to run the program
if __name__ == "__main__":
    main()