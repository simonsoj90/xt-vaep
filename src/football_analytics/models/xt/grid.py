import numpy as np

class XTGrid:
    def __init__(self, n_x: int = 16, n_y: int = 12, gamma: float = 0.97, max_iter: int = 200, tol: float = 1e-6, pitch_w: float = 120.0, pitch_h: float = 80.0):
        self.n_x=n_x
        self.n_y=n_y
        self.gamma=gamma
        self.max_iter=int(max_iter)
        self.tol=tol
        self._w=float(pitch_w)
        self._h=float(pitch_h)
        self.v=None

    def _bin(self,x,y):
        xv=np.asarray(x,dtype=float)
        yv=np.asarray(y,dtype=float)
        xv=np.nan_to_num(xv,nan=0.0,posinf=self._w,neginf=0.0)
        yv=np.nan_to_num(yv,nan=0.0,posinf=self._h,neginf=0.0)
        xi=np.floor((xv/self._w)*self.n_x)
        yi=np.floor((yv/self._h)*self.n_y)
        xi=np.clip(xi,0,self.n_x-1).astype(int)
        yi=np.clip(yi,0,self.n_y-1).astype(int)
        return xi,yi

    def value(self,x,y):
        if self.v is None:
            raise RuntimeError("grid not initialized")
        xi,yi=self._bin(x,y)
        return self.v[yi,xi]

    def _smooth(self,arr,passes=2):
        a=arr.copy()
        for _ in range(passes):
            pad=np.pad(a,((1,1),(1,1)),"edge")
            up=pad[:-2,1:-1]
            down=pad[2:,1:-1]
            left=pad[1:-1,:-2]
            right=pad[1:-1,2:]
            diag1=pad[:-2,:-2]
            diag2=pad[:-2,2:]
            diag3=pad[2:,:-2]
            diag4=pad[2:,2:]
            neigh=(up+down+left+right)*2 + (diag1+diag2+diag3+diag4)*1
            a=0.6*a + 0.4*(neigh/12.0)
        return a

    def fit(self, events):
        shots=events[events["event_type"].astype(str).eq("Shot")]
        if len(shots)==0:
            self.v=np.zeros((self.n_y,self.n_x),dtype=float)
            return
        xi,yi=self._bin(shots["x"].to_numpy(),shots["y"].to_numpy())
        is_goal=shots["event_outcome"].astype(str).str.contains("goal",case=False,na=False).to_numpy().astype(int)
        cnt=np.zeros((self.n_y,self.n_x),dtype=float)
        gls=np.zeros((self.n_y,self.n_x),dtype=float)
        for j in range(len(xi)):
            cnt[yi[j],xi[j]]+=1.0
            gls[yi[j],xi[j]]+=is_goal[j]
        with np.errstate(divide="ignore",invalid="ignore"):
            v=np.where(cnt>0, gls/cnt, 0.0)
        v=self._smooth(v,passes=3)
        self.v=v

    def set_grid(self,v):
        v=np.array(v,copy=True)
        self.v=v
        self.n_y,self.n_x=v.shape

    def save(self,path):
        np.savez(path,v=self.v)

    @staticmethod
    def load(path):
        z=np.load(path,allow_pickle=True)
        return z["v"]
