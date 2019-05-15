from time import time

from predict_func import get_predict_args, make_a_prediction

def main():
    # Command Line Args
    in_args = get_predict_args()
    
    device = 'cuda' if in_args.gpu else 'cpu'
    
    # Start Time
    start_time = time()
    
    # Predict
    make_a_prediction(
        checkpoint=in_args.checkpoint, 
        image_path=in_args.image_path,
        device=device, 
        topk=in_args.top_k, 
        cat_to_name=in_args.category_names, 
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