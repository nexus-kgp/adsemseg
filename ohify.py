import cv2 as cv



#This is collected frm color_map.py

 temp= {(0, 0, 0): 0,
 (0, 0, 64): 8,
 (0, 0, 128): 1,
 (0, 0, 192): 9,
 (0, 64, 0): 16,
 (0, 64, 128): 17,
 (0, 128, 0): 2,
 (0, 128, 64): 10,
 (0, 128, 128): 3,
 (0, 128, 192): 11,
 (0, 192, 0): 18,
 (0, 192, 128): 19,
 (128, 0, 0): 4,
 (128, 0, 64): 12,
 (128, 0, 128): 5,
 (128, 0, 192): 13,
 (128, 64, 0): 20,
 (128, 128, 0): 6,
 (128, 128, 64): 14,
 (128, 128, 128): 7,
 (128, 128, 192): 15}


def ohify(img):
    cl = np.zeros((img.shape[0],img.shape[1],21))
    for j,row in enumerate(img):
        for i,col in enumerate(row):
            try:
                # cl[row][col][np.where(np.all(maps==col,axis=1))[0][0]] = 1
                cl[j][i][temp[tuple(col.tolist())]] = 1
            except KeyError as e:
                pass
    return cl

def ohify_and_save(path):
    for file in os.listdir(path):
        if file.endswith('png'):
            try:
                img=cv.imread(file)
            except Exception as e:
                print(e)
            oheified = ohify(img)
            np.savez_compressed("./ohe/{}".format(file),oheified)


def ohify_labels(labels):
    cl = np.zeros((256,256,21))
    for j,row in enumerate(labels):
        for i,col in enumerate(row):
            # cl[row][col][np.where(np.all(maps==col,axis=1))[0][0]] = 1
            cl[j][i][col.data[0]] = 1
    return cl




