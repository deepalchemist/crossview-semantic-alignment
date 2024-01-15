

# Last Change:  2024-01-12 06:43:26

import os
import cv2
import math
import numpy as np
from pathlib import Path
from itertools import islice
from matplotlib import colors
from PIL import Image, ImageDraw, ImageFont

class cv2_helper(object):
    
    @staticmethod
    def init():
        return
    
    @staticmethod
    def read_image(fpath):
        # IMREAD_COLOR: always convert image to the 3 channel BGR color image.
        try:
            img = cv2.imread(fpath, cv2.IMREAD_COLOR) # [0 255], BGR, HW3, numpy array
            assert img is not None, print("cv2.imread returns None!")
            return img
        except Exception as e:
            print(f"Invalid image path: {fpath}")
            print(e)
            return None
    
    @staticmethod
    def write_image(img, fpath, overwrite=False):
        if os.path.isfile(fpath) and (not overwrite):
            return 
        dirname = os.path.dirname(fpath)
        Path(dirname).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(fpath, img)
        
    @staticmethod
    def resize(img, shorter_size=None, longer_size=None, dsize=None):
        ''' Resize image via shorter_size or longer_size
        Args:
          img: numpy array with shape [im_h, im_w, 3]
          shorter_size(Int): min number of pixel
          dsize: (dh, dw)
        '''
        h,w,c = img.shape
        if dsize is not None:
            dh, dw = dsize
            return cv2.resize(img, dsize=(dw, dh), interpolation=cv2.INTER_LINEAR)

        if h>=w:
            if shorter_size is not None:
                dw = shorter_size
                dh = int(h/w * shorter_size)
            else:
                assert longer_size is not None
                dh = longer_size
                dw = int(longer_size/h * w)
        else:
            if shorter_size is not None:
                dh = shorter_size
                dw = int(w/h * shorter_size)
            else:
                assert longer_size is not None
                dw = longer_size
                dh = int(longer_size/w * h)
            
        img = cv2.resize(img, dsize=(dw, dh), interpolation=cv2.INTER_LINEAR)
        return img 
        
    @staticmethod
    def pad_around_image(img, 
                         pad_width=(5,5,0,0), 
                         pad_value=255.):
        """ Add color border around an image. The resulting image size is not changed.
        Args:
          img: numpy array with shape [im_h, im_w, 3]
          pad_width: (top, bottom, left, right), measured in pixel
          pad_value: scalar, or numpy array with shape [3]; the color of the border
        Returns:
          rst_img: numpy array with shape [im_H, im_W, 3]
        """
        assert (img.ndim == 3) and (img.shape[2] == 3)
        img = np.copy(img)
    
        if isinstance(pad_value, np.ndarray):
            # reshape to [1, 1, 3]
            pad_value = pad_value.flatten()[np.newaxis, np.newaxis, :]
        
        top, bottom, left, right = pad_width
        h, w = img.shape[0], img.shape[1]
        H = h + top + bottom
        W = w + left + right
        
        rst_img = (np.ones([H, W, 3]) * pad_value).astype(img.dtype)
        
        rst_img[:top, :, :] = pad_value
        rst_img[-bottom:, :, :] = pad_value
        rst_img[:, :left, :] = pad_value
        rst_img[:, -right:, :] = pad_value
        rst_img[top:top+h, left:left+w, :] = img
        
        return rst_img
    

    @staticmethod
    def make_image_row(img_paths=None, 
                       img_titles=None,
                       imgs=None, 
                       dsize=(100, 100), # (dst_h, dst_w)
                       pad_value=255.):
        """ Make a row of images with space in between.
        Args:
          img_paths: a list of image paths
          img_titles: a list of image titles
          imgs: a list of [h, w, 3], BGR images
          dsize: resize shape (dst_h, dst_w)
          pad_val: scalar, or numpy array with shape [3]; the color of the space
        Returns:
          rst_img: a numpy array with shape [h, W, 3]
        """
        
        if imgs is None:
            assert img_paths is not None
            imgs = [cv2_helper.read_image(p) for p in img_paths]
            # imgs = [x.transpose((2, 0, 1)) for x in imgs] # transpose HW3 to 3HW
        
        imgs = [cv2.resize(x, dsize=dsize[::-1], interpolation=cv2.INTER_LINEAR) for x in imgs]
        
        n_cols = len(imgs)
        h, w = dsize[0], dsize[1]
        
        k_space = 7  # means q_g space
        space = 2 # means g_g space

        H = h
        W = w * n_cols + space * (n_cols - 2) + k_space * space
        
        if isinstance(pad_value, np.ndarray):
            # reshape to [1, 1, 3]
            pad_value = pad_value.flatten()[np.newaxis, np.newaxis, :]
            
        rst_img = (np.ones([H, W, 3]) * pad_value).astype(imgs[0].dtype)

        rst_img[0:h, 0:w, :] = imgs[0]  # query image

        start_w = w + k_space * space
        for im in imgs[1:]:
            end_w = start_w + w
            rst_img[0:h, start_w:end_w, :] = im
            start_w = end_w + space
        
        if img_titles is None:
            return rst_img
        
        # Plot title on images
        assert len(imgs) == len(img_titles)
        font_scale = 0.3 * (dsize[0]/100.)  # 2 is a scale param
        
        pad_top = int(h * 0.1)
        rst_img = cv2_helper.pad_around_image(rst_img, (pad_top, 0, 0, 0))
        
        for idx, title in enumerate(img_titles):
            if idx==0:
                org = (0, pad_top-30)
            elif idx==1:
                org = (w+k_space*space, pad_top-30)
            else:
                org = (w+k_space*space+(idx-1)*(w+space), pad_top-30)
           
            # Plot text
            rst_img = Image.fromarray(cv2.cvtColor(rst_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(rst_img)
            fontText = ImageFont.truetype("./simsun.ttc", 30, encoding="utf-8")
            draw.text(org, f"{title}", (0, 0, 0), font=fontText)
            rst_img = cv2.cvtColor(np.asarray(rst_img), cv2.COLOR_RGB2BGR)
            #rst_img = cv2.putText(rst_img, f"{title}", org=org, fontScale=font_scale, 
            #                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0),
            #                      thickness=1, lineType=cv2.LINE_AA)
            
        return rst_img
    
    @staticmethod
    def make_image_grid(img_paths=None, 
                        img_titles=None,
                        imgs=None, 
                        n_cols=1, 
                        dsize=(100, 100), # (dst_h, dst_w)
                        pad_value=255.):
        """ Make a grid of images with space in between.
        Args:
          img_paths: a list of image paths
          img_titles: a list of image titles
          imgs: a list of [h, w, 3], BGR images
          n_cols: number images per row
          dsize: resize shape (dst_h, dst_w)
          pad_value: scalar, or numpy array with shape [3]; the color of the space
        Returns:
          rst_img: a numpy array with shape [H, W, 3]
        """
        if imgs is None:
            assert img_paths is not None
            imgs = [cv2_helper.read_image(p) for p in img_paths]
            # imgs = [x.transpose((2, 0, 1)) for x in imgs] # transpose HW3 to 3HW
            
        if img_titles is not None:
            assert len(imgs) == len(img_titles)
          
        # List of length in which we have to split
        assert len(imgs) > 0, print("Length of images must > 0")
        
        tot_num = len(imgs)
        if tot_num <= n_cols:
            return cv2_helper.make_image_row(None, img_titles, imgs, dsize, pad_value)
            
        n_rows = tot_num // n_cols
        mod = tot_num % n_cols
        length_to_split = [n_cols] * n_rows
        if mod != 0:
            n_rows += 1
            length_to_split.append(mod)
            
        # Using islice
        imgs = iter(imgs)
        imgs = [list(islice(imgs, elem)) for elem in length_to_split]
        if img_titles is not None:
            img_titles = iter(img_titles)
            img_titles = [list(islice(img_titles, elem)) for elem in length_to_split]
        
        # Make each row
        rst_imgs = []
        for idx, img in enumerate(imgs):
            title = img_titles[idx] if img_titles is not None else None
            rst_img = cv2_helper.make_image_row(None, title, img, dsize, pad_value)
            rst_imgs.append(rst_img)
        
        # Concat white images
        if mod != 0:
            #print(f"n_cols: {n_cols} cannot divide image number: {tot_num}, cat white images.")
            H, W = rst_imgs[0].shape[0], rst_imgs[0].shape[1]
            last_img = np.ones([H, W, 3]).astype(rst_imgs[-1].dtype)*pad_value
            last_img_w = rst_imgs[-1].shape[1]
            last_img[:, 0:last_img_w, :] = rst_imgs[-1]
            # pop and append
            rst_imgs.pop()
            rst_imgs.append(last_img)
            
        rst_img = np.concatenate(rst_imgs, axis=0)
        return rst_img
    
    @staticmethod
    def _make_image_grid(img_paths=None, 
                        img_titles=None,
                        imgs=None, 
                        n_rows=1, 
                        dsize=(100, 100), # (dst_h, dst_w)
                        pad_value=255.):
        """ Make a grid of images with space in between.
        Args:
          img_paths: a list of image paths
          img_titles: a list of image titles
          imgs: a list of [h, w, 3], BGR images
          n_rows: number row of image grid 
          dsize: resize shape (dst_h, dst_w)
          pad_value: scalar, or numpy array with shape [3]; the color of the space
        Returns:
          rst_img: a numpy array with shape [H, W, 3]
        """
        if imgs is None:
            assert img_paths is not None
            imgs = [cv2_helper.read_image(p) for p in img_paths]
            # imgs = [x.transpose((2, 0, 1)) for x in imgs] # transpose HW3 to 3HW
            
        if img_titles is not None:
            assert len(imgs) == len(img_titles)
          
        # List of length in which we have to split
        tot_num = len(imgs)
        n_cols = tot_num // n_rows
        mod = tot_num % n_rows
        length_to_split = [n_cols] * n_rows
        if mod != 0:
            n_cols += 1
            length_to_split.append(mod)
            
        # Using islice
        imgs = iter(imgs)
        imgs = [list(islice(imgs, elem)) for elem in length_to_split]
        img_titles = iter(img_titles)
        img_titles = [list(islice(img_titles, elem)) for elem in length_to_split]
        
        # Make each row
        rst_imgs = []
        for img, title in zip(imgs, img_titles):
            rst_img = cv2_helper.make_image_row(None, title, img, dsize, pad_value)
            rst_imgs.append(rst_img)
        
        # Concat white images
        if mod != 0:
            #print(f"n_rows: {n_rows} cannot divide image number: {tot_num}, cat white images.")
            H, W = rst_imgs[0].shape[0], rst_imgs[0].shape[1]
            last_img = np.zeros([H, W, 3]).astype(rst_imgs[-1].dtype)
            last_img_w = rst_imgs[-1].shape[1]
            last_img[:, 0:last_img_w, :] = rst_imgs[-1]
            # pop and append
            rst_imgs.pop()
            rst_imgs.append(last_img)
            
        rst_img = np.concatenate(rst_imgs, axis=0)
        return rst_img
    
    @staticmethod
    def plot_bbox_on_image(bbox, img_path=None, img=None, bbox_title=None):
        """
        Args:
            imgs(Array): [h, w, 3] BGR
            bbox(List[List]): float location ranges from 0 to 1, [[left, top, right, bottom], ...]
            bbox_title(List): [0.7, ...]
        """
        if img is None:
            assert img_path is not None
            img = cv2_helper.read_image(img_path) # HW3 BGR
        assert img.shape[2]==3 and img.ndim==3
        img_h, img_w = img.shape[:2]
        
        if bbox_title is not None:
            assert len(bbox_title)==len(bbox)
            
        bbox_color = [t*255. for t in colors.to_rgb('royalblue')][::-1] # RGB to BGR
        title_color = bbox_color
        for idx in range(len(bbox)):
            if len(bbox[idx]) != 4:  # Skip bug bbox
                continue
            x1,y1,x2,y2 = bbox[idx]
            x1,y1,x2,y2 = math.ceil(x1*img_w), math.ceil(y1*img_h), math.floor(x2*img_w), math.floor(y2*img_h)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=2)
            
            if bbox_title is not None:
                # thickness=negative meansfilled rectangle
                title_h, title_w = 20, 45
                img = cv2.rectangle(img, (x1, y1), (x1+title_w, y1+title_h),title_color,thickness=-1)
                title = bbox_title[idx]
                if isinstance(title, float) or isinstance(title, int):
                    title = "{:.2f}".format(title)
                print("->", title)
                img = cv2.putText(img, title, org=(x1+4, y1+15), 
                                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                                  color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        return img
    
    @staticmethod
    def plot_attnmap_on_image(attn_map, img=None, img_path=None, out_size=(256, 128)):
        '''
        Args:
            img: cv2 image, BGR, 0-255
            attn_map: target attention map (grayscale) 0-255
            out_size: desired output image size (height width)

        Returns:
            numpy array of shape HWC, 0-255, uint8
        '''
        if img is None:
            assert img_path is not None
            img = cv2_helper.read_image(img_path) # HW3 BGR
        assert img.shape[2]==3 and img.ndim==3
        
        org_im = cv2.resize(img, (out_size[1], out_size[0]))
        activation_map = cv2.resize(attn_map, (out_size[1], out_size[0]), interpolation=cv2.INTER_CUBIC)

        activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
        # Heatmap on picture
        im_with_heatmap = np.float32(activation_heatmap) + np.float32(org_im)
        max_val = np.clip(np.max(im_with_heatmap), a_min=1e-10, a_max=None)
        im_with_heatmap = im_with_heatmap / max_val
        im_with_heatmap = im_with_heatmap[..., ::-1]  # BGR to RGB
        im_with_heatmap = np.uint8(255 * im_with_heatmap)
        return im_with_heatmap 

    @staticmethod
    def video_to_frame(video_path, savedir, fix_frame=0, shorter_size=-1):
        vidcap = cv2.VideoCapture(video_path)
        
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fps = round(fps) if fps>=1 else 1
        nbframes = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        tgt_frame_idx = list(range(0, nbframes))
        if fix_frame > 0:
            space = nbframes // fix_frame
            tgt_frame_idx = list(range(0, nbframes, space))
            if len(tgt_frame_idx) > fix_frame:
                tgt_frame_idx = tgt_frame_idx[:fix_frame]

        success, image = vidcap.read()
        import pdb; pdb.set_trace()
        if shorter_size != -1 and success:
            image = cv2_helper.resize(image, shorter_size)
        cnt = -1
        valid_cnt = 0
        while success and cnt<nbframes:
            cnt += 1
            if cnt in tgt_frame_idx:
                fpath = os.path.join(savedir, f"frame_{valid_cnt}.jpg")
                cv2_helper.write_image(image, fpath) # save frame as JPEG file      
                valid_cnt += 1
            success, image = vidcap.read()
            if shorter_size != -1 and success:
                image = cv2_helper.resize(image, shorter_size)
            # print('read a new frame: ', success)
            
        rst = {"fps": fps, "tot_frame": nbframes}   
        return rst
    
    @staticmethod
    def frame_to_video(frame_dir, video_path, fps=30):
        """
        Args:
            frame_dir: frame_0.jpg
            video_path: a/b/xxx.mp4
        """
        frame_array = []
        files = [f for f in os.listdir(frame_dir) if f.endswith(".jpg")]
        
        # for sorting the file names properly
        files.sort(key = lambda x: int(x[6:-4]))
        
        for i in range(len(files)):
            filename = os.path.join(frame_dir, files[i])
            
            # reading each files
            img = cv2_helper.read_image(filename)
            height, width, layers = img.shape
            size = (width, height)
        
            # inserting the frames into an image array
            frame_array.append(img)
            
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()
        
        return
