import os
import time
from PaddleOCR.paddleocr import PaddleOCR, draw_ocr


#  python ./tools/infer/predict_system.py --image_dir=C:/Users/shinelon/Desktop/test.jpg --det_model_dir=./inference/ch_ppocr_mobile_v1.1_det_infer  --rec_model_dir=./inference/ch_ppocr_mobile_v1.1_rec_infer  --cls_model_dir=./inference/ch_ppocr_mobile_v1.1_cls_infer  --use_angle_cls=True  --use_space_char=True  --use_gpu=False

if __name__ == '__main__':

    PATH_IMG_IN = './in'
    filename = os.path.join(PATH_IMG_IN, '1.png')
    filename = 'C:/Users/shinelon/Desktop/test1.png'


    cls_model_dir = "./ORC/PaddleOCR/inference/cls"
    det_model_dir= "./ORC/PaddleOCR/inference/2.1/det/ch"
    rec_model_dir= "./ORC/PaddleOCR/inference/2.1/rec/ch"
    ocr = PaddleOCR(cls_model_dir = cls_model_dir, det_model_dir= det_model_dir, rec_model_dir= rec_model_dir) # need to run only once to download and load model into memory
    start = time.perf_counter()
    result = ocr.ocr(filename, rec=True)
    end = time.perf_counter()
    print('检测文字区域 耗时{}'.format(end-start))
    for line in result:
        print(line)
    #每个矩形，从左上角顺时针排列

    # for rect1 in rects:
    #     print(rect1)


# from PIL import Image
# image = Image.open(filename).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('C:/Users/shinelon/Desktop/testresult1.jpg')
