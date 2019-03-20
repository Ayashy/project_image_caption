import argparse
import preprocess


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Caption generation model')

    parser.add_argument('-task', '-t', help='path to image',type=str, required=True)
    parser.add_argument('--dataset' , help='Name of the dataset to use [flickr,coco]')
    parser.add_argument('--target',help='Directory where to store the processed data')
    parser.add_argument('--source',help='Directory that contains raw data')

    args = parser.parse_args()


    if args.task.lower()=='preprocess':
        print('------------ Executing preprocessing mode ------------')
        if args.dataset is not None:
            if args.dataset.lower()=='flickr':
                preprocess.preprocess_flickr_data(source=args.source,target=args.target)
            if args.dataset.lower()=='coco':
                preprocess.preprocess_coco_data(source=args.source,target=args.target)
        
    if args.task.lower()=='train':
        print('------------ Executing training mode ------------')
    
    if args.task.lower()=='test':
        print('------------ Executing testing mode ------------')

    if args.task.lower()=='predict':
        print('------------ Executing prediction mode ------------')
