"""No-learned-features, multi-scale raw grayscale normalized correlation."""
import cv2
import numpy as np

class RawTemplate:
    max_side=384
    scales=(.10,.16,.25,.40,.63,1.)
    aspect_factors=(.75,1.,4/3)

    def enroll(self,rgb):
        return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)

    def encode_query(self,rgb):
        factor=min(1.,self.max_side/max(rgb.shape[:2]))
        resized=cv2.resize(rgb,(max(1,round(rgb.shape[1]*factor)),max(1,round(rgb.shape[0]*factor))),interpolation=cv2.INTER_AREA)
        return {'gray':cv2.cvtColor(resized,cv2.COLOR_RGB2GRAY).astype(np.float32),
                'scale_x':resized.shape[1]/rgb.shape[1],'scale_y':resized.shape[0]/rgb.shape[0]}

    def predict(self,bank,query):
        if isinstance(query,np.ndarray): query=self.encode_query(query)
        gray=query['gray']; h,w=gray.shape; aspect=bank.shape[1]/bank.shape[0]
        best={'score':-1.,'box':[0.,0.,w/query['scale_x'],h/query['scale_y']]}
        shapes=set()
        for scale in self.scales:
            for a in self.aspect_factors:
                side=min(h,w)*scale; ar=aspect*a
                th=max(3,round(side/np.sqrt(ar))); tw=max(3,round(side*np.sqrt(ar)))
                if th>h or tw>w or (th,tw) in shapes: continue
                shapes.add((th,tw))
                template=cv2.resize(bank,(tw,th),interpolation=cv2.INTER_AREA)
                if template.std()<1e-6: continue
                response=cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
                _,score,_,(x,y)=cv2.minMaxLoc(response)
                if not np.isfinite(score): raise ValueError('Nonfinite raw template correlation')
                if score>best['score']:
                    best={'score':float(score),'box':[x/query['scale_x'],y/query['scale_y'],
                          (x+tw)/query['scale_x'],(y+th)/query['scale_y']]}
        return [best]

    def assert_unchanged(self): pass
    def provenance(self):
        return {'name':'raw_template','borrowed_pretrained_weights':False,'trainable_parameters':0,
                'support_images_per_target':1,'descriptor':'Raw grayscale pixels',
                'score':'Zero-mean normalized cross correlation, maximum across fixed sizes/aspects',
                'max_side':self.max_side,'scales':self.scales,'aspect_factors':self.aspect_factors,
                'opencv':cv2.__version__,'device':'cpu','weight_updates':0}
