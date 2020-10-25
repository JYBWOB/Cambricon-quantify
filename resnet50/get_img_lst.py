import os

img_path = './data/imgs/'
save_path = './data/image.lst'

def get_img_lst(root, savepath):
    with open(savepath, 'w') as f:
        for file in os.listdir(root):  
            f.write(img_path + file + '\n')

if __name__ == '__main__':
    get_img_lst(img_path, save_path)
