import cv2
import copy
import numpy as np
import seaborn as sns
import imageio
import matplotlib.pyplot as plt


class Drawer(object):
    def __init__(self, color=(255, 255, 0), font=cv2.FONT_HERSHEY_DUPLEX):
        self.color = color
        self.color_palette = np.array(sns.color_palette("hls", 8)) * 255
        self.RED = (0, 0, 255)
        self.LESSRED = (0, 20, 100)
        self.TEAL = (148, 184, 0)
        self.DARKPURPLE = (17, 7, 60)
        self.font = font
        self.fontScale = 1
        self.fontThickness = 1
        self.indFontScale = self.fontScale * 0.5
        self.indFontThickness = self.fontThickness * 2
        self.indTextSize = cv2.getTextSize(text=str(
            '1'), fontFace=self.font, fontScale=self.indFontScale, thickness=self.indFontThickness)[0]

    def _resize(self, frame):
        height, width = frame.shape[:2]
        if height != self.frameHeight:
            scale = float(height) / self.frameHeight
            frame = cv2.resize(
                frame, (int(width / scale), int(self.frameHeight)))
        return frame

    def _put_label(self, frame, label):
        border = 3
        cv2.putText(frame, label, (border, 20+border),
                    self.font, self.fontScale,
                    self.color, self.fontThickness)

    def draw_chosen(self, frameDC, track):
        if track:
            msg = 'FOLLOWING: {}'.format(track.track_id)
            color = (0, 255, 0)
        else:
            msg = 'FOLLOWING: NIL'
            color = self.color
        fontScale = 1.2
        fontThickness = 3
        cv2.putText(
            frameDC, msg, (frameDC.shape[1]-310, 10+24), self.font, fontScale, color, fontThickness)

    def draw_bbs(self, frameDC, bbs):
        if bbs is None or len(bbs) == 0:
            return
        frame_h, frame_w = frameDC.shape[:2]
        for i, bb in enumerate(bbs):
            _color = self.color_palette[i]
            print('Color: ', _color)
            if bb is None:
                continue
            l, t, w, h = int(bb[0]), int(bb[1]), int(
                bb[2]), int(bb[3])  # [int(x) for x in bb[0]]
            r = l + w - 1
            b = t + h - 1

            cv2.rectangle(frameDC, (l, t), (r, b), _color, 2)
        return frameDC

    def draw_bbs_w_fine_grain(self, frameDC, bbs, top_ns, label=''):
        if bbs is None or len(bbs) == 0:
            return
        # frameDC = copy.deepcopy(frame)
        frame_h, frame_w = frameDC.shape[:2]
        for i, bb in enumerate(bbs):
            _color = self.color_palette[i]
            print('Color: ', _color)
            if bb is None:
                continue
            l, t, w, h = int(bb[0]), int(bb[1]), int(
                bb[2]), int(bb[3])  # [int(x) for x in bb[0]]
            r = l + w - 1
            b = t + h - 1

            cv2.rectangle(frameDC, (l, t), (r, b), _color, 2)

            for count, j in enumerate(top_ns[i]):
                text = '{}:{}'.format(j[0], j[1])

                cv2.putText(frameDC,
                            text,
                            (l+5, b-len(top_ns) * (30) + (30)*(count + 1)),
                            self.font, self.fontScale, _color, self.fontThickness)
        self._put_label(frameDC, label)

    def draw_label(self, frame, label=''):
        frameDC = copy.deepcopy(frame)
        self._put_label(frameDC, label)
        return frameDC

    # det = ( class, confidence , (x, y, w, h) )
    def draw_dets(self, frame, dets, color=None, label=''):
        if dets is None or len(dets) == 0:
            return frame
        #if color is None:
            #color = self.color
        frameDC = copy.deepcopy(frame)
        self._put_label(frameDC, label)
        for i, det in enumerate(dets):
            _color = self.color_palette[i%8]
            l, t, w, h = int(det[2][0]), int(det[2][1]), int(det[2][2]), int(det[2][3])
            r = l + w - 1
            b = t + h - 1
            text = '{}: {:0.2f}%'.format(det[0], det[1]*100)
            cv2.rectangle(frameDC, (l, t), (r, b), _color, 2)
            cv2.putText(frameDC, text, (l+5, b-10), self.font, self.fontScale, _color, self.fontThickness)
        return frameDC

    def draw_dets_video(self, frame, dets, infer_time, color=None, label=''):
        if dets is None or len(dets) == 0:
            return frame
        #if color is None:
            #color = self.color
        frameDC = copy.deepcopy(frame)
        self._put_label(frameDC, label)
        infer_text = 'Inference Time per Frame: {:0.2f} ms'.format(infer_time * 1000)
        for i, det in enumerate(dets):
            _color = self.color_palette[i%8]
            l, t, w, h = int(det["bbox"][0]), int(det["bbox"][1]), int(det["bbox"][2]), int(det["bbox"][3])
            r = l + w - 1
            b = t + h - 1
            text = '{}: {:0.2f}%'.format(det["category_id"], det["score"]*100)
            cv2.rectangle(frameDC, (l, t), (r, b), _color, 2)
            cv2.putText(frameDC, text, (l+5, b-10), self.font, self.indFontScale, _color, self.fontThickness)    
        cv2.putText(frameDC, infer_text, (10, 30), self.font, self.fontScale * 0.75, (0, 0, 255), self.indFontThickness) 
        return frameDC

    def draw_bb_name(self, frame, bb, name, color=None, label=''):
        if color is None:
            color = self.color
        frameDC = copy.deepcopy(frame)
        frame_h, frame_w = frame.shape[:2]
        l = max(0, int(bb['rect']['l']))
        t = max(0, int(bb['rect']['t']))
        r = min(frame_w-1, int(bb['rect']['r']))
        b = min(frame_h-1, int(bb['rect']['b']))
        text = str('{}'.format(name))
        cv2.rectangle(frameDC, (l, t), (r, b), color, 2)
        cv2.putText(frameDC,
                    text,
                    (l+5, b-10),
                    self.font, self.fontScale, color, self.fontThickness)
        self._put_label(frameDC, label)
        return frameDC

if __name__ == "__main__":
    # od = get_OD()
    drawer = Drawer()
    # bbs = od.detect_ltwh('000000466319.jpg', classes=['car'], buffer=0.3)
    # img_with_bb = drawer.draw_bbs(imageio.imread('000000466319.jpg'), [[199.02, 86.78, 135.59, 339.53]])
    #img_with_bb_det = drawer.draw_dets(imageio.imread('000000000013.jpg'), [('person', 0.51, (156.64, 568.61, 335.65, 706.84))])
    #img_with_bb_det = drawer.draw_dets(frame, )
    plt.imshow(img_with_bb_det)
    plt.show()
