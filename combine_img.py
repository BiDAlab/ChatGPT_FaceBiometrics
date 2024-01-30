import pandas as pd
import glob as glob
import cv2
import numpy as np
import os
import argparse

def main(args):  
    # Creating folders if dont exists
    print('Creating folders if dont exists...')
    if not os.path.exists(args.path_save_matrix):
        os.makedirs(args.path_save_matrix)
    if not os.path.exists(args.path_save_combined):
        os.makedirs(args.path_save_combined)

    # Read Data
        print('Reading data...')
    df = pd.read_csv(args.df_name, sep=',')
    #################### Combine images 1 vs 1 ####################
    print('Combining images...')
    width, height = 256, 256
    combined_img = []
    for i, row in df.iterrows():
        image1 = cv2.imread(os.path.join(args.append_path, row['img1']))
        image2 = cv2.imread(os.path.join(args.append_path, row['img2']))
        image1 = cv2.resize(image1, (width, height)) 
        image2 = cv2.resize(image2, (width, height))

        combined_image = np.hstack((image1, image2))
        cv2.imwrite(os.path.join(args.path_save_combined, str(i)+'.jpg'), combined_image)
        print(os.path.join(args.path_save_combined, str(i)+'.jpg'))
        combined_img.append(os.path.join(args.path_save_combined, str(i)+'.jpg'))
    df['combined_img'] = combined_img
    #################### Combine images Matrix ####################
    print('Combining images matrix...')
    cols = 4 
    rows = 3 

    border_w = 15
    border_top = 25

    matrix = []
    matrix_pos = []

    count = 0

    while count * (rows * cols) < len(df):
        final_img = None
        for i in range(rows):
            row_img = None
            for j in range(cols):
                row_i_aux = cols*i + j
                row_i = row_i_aux + (rows * cols) * count 
                print(row_i)
                if row_i >= len(df):
                    break
                
                img = cv2.imread(df['combined_img'][row_i])
                
                h, w, c = img.shape
                canvas = np.ones((h + border_top, w, c))
                canvas[border_top:, :, :] = img
                matrix_pos.append(row_i_aux)
                matrix.append(count)
                canvas = cv2.putText(canvas, str(row_i_aux), (int(0.49*img.shape[1]), 23), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255])
                if row_img is None:
                    row_img = canvas
                    row_img = cv2.copyMakeBorder(row_img, top=0, bottom=0, left=border_w, right=0, borderType=cv2.BORDER_CONSTANT, value=[255, 0, 0])
                else:
                    row_img = np.hstack((row_img, canvas))
                    # Add border after hstack
                row_img = cv2.copyMakeBorder(row_img, top=0, bottom=0, left=0, right=border_w, borderType=cv2.BORDER_CONSTANT, value=[255, 0, 0])

            if final_img is None:
                final_img = row_img
                final_img = cv2.copyMakeBorder(final_img, top=border_w, bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[255, 0, 0])
            else:
                if row_img is not None:
                    final_img = np.vstack((final_img, row_img))
            final_img = cv2.copyMakeBorder(final_img, top=0, bottom=border_w, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[255, 0, 0])

        cv2.imwrite(os.path.join(args.path_save_matrix, f'{count:02}.jpg'), final_img)
        
        count += 1

    print('Saving data...')
    df['matrix'] = matrix
    df['matrix_pos'] = matrix_pos
    df.to_csv(args.df_name.split('.')[0] + '_matrixpos.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combinig faces')
    # Set Params
    parser.add_argument('--df_name', required=True)
    parser.add_argument('--path_save_matrix', default=r'BBDD\matrix_preprocess')
    parser.add_argument('--path_save_combined', default=r'BBDD\combined_preprocess')
    parser.add_argument('--append_path', default='')

    args = parser.parse_args()
    main(args)
