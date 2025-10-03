import numpy as np
import pandas as pd

class XTGrid:
    def __init__(self,n_x=16,n_y=12,max_iter=300,tol=1e-8,gamma=0.97):
        self.n_x=n_x
        self.n_y=n_y
        self.max_iter=max_iter
        self.tol=tol
        self.gamma=gamma
        self.v=np.zeros((n_y,n_x))
        self.p_shot=np.zeros((n_y,n_x))
        self.p_goal=np.zeros((n_y,n_x))
        self.M=None
        self._w=120.0
        self._h=80.0

    def _bin_valid(self,x,y):
        x=np.asarray(x)
        y=np.asarray(y)
        m=np.isfinite(x)&np.isfinite(y)
        xi=np.full(x.shape,-1,dtype=int)
        yi=np.full(y.shape,-1,dtype=int)
        if m.any():
            xs=np.clip((x[m]/self._w)*self.n_x,0,self.n_x-1).astype(int)
            ys=np.clip((y[m]/self._h)*self.n_y,0,self.n_y-1).astype(int)
            xi[m]=xs
            yi[m]=ys
        return xi,yi,m
    
    def _bin(self,x,y):
        xi,yi,_=self._bin_valid(x,y)
        return xi,yi

    def fit(self,events:pd.DataFrame,pitch_w=120.0,pitch_h=80.0):
        self._w=float(pitch_w); self._h=float(pitch_h)
        ex=events.copy()
        for c in ["x","y","pass_end_x","pass_end_y","carry_end_x","carry_end_y"]:
            if c in ex.columns:
                ex[c]=pd.to_numeric(ex[c],errors="coerce")
        xi,yi,m=self._bin_valid(ex.get("x"),ex.get("y"))
        total=np.zeros((self.n_y,self.n_x))
        if m.any():
            for a,b in zip(yi[m],xi[m]):
                total[a,b]+=1
        s=np.zeros((self.n_y,self.n_x))
        h=np.zeros((self.n_y,self.n_x))
        sh=ex[ex["event_type"].astype(str).str.lower().eq("shot")]
        if len(sh)>0:
            xs,ys,ms=self._bin_valid(sh["x"],sh["y"])
            goalmask=sh["event_outcome"].astype(str).str.contains("goal",case=False,na=False).to_numpy()
            if ms.any():
                for a,b in zip(ys[ms],xs[ms]):
                    s[a,b]+=1
                for a,b,u in zip(ys[ms],xs[ms],goalmask[ms]):
                    if u: h[a,b]+=1
        self.p_shot=np.divide(s,total,where=total>0)
        self.p_goal=np.divide(h,s,where=s>0)
        ex["end_x"]=np.where(ex["event_type"].astype(str).eq("Pass"),ex.get("pass_end_x"),np.where(ex["event_type"].astype(str).eq("Carry"),ex.get("carry_end_x"),np.nan))
        ex["end_y"]=np.where(ex["event_type"].astype(str).eq("Pass"),ex.get("pass_end_y"),np.where(ex["event_type"].astype(str).eq("Carry"),ex.get("carry_end_y"),np.nan))
        mv=ex[ex["event_type"].isin(["Pass","Carry"]) & ex["end_x"].notna() & ex["end_y"].notna()]
        k=self.n_x*self.n_y
        M=np.zeros((k,k))
        if len(mv)>0:
            x0,y0,m0=self._bin_valid(mv["x"],mv["y"])
            x1,y1,m1=self._bin_valid(mv["end_x"],mv["end_y"])
            mm=m0 & m1
            if mm.any():
                idx=y0[mm]*self.n_x+x0[mm]
                jdx=y1[mm]*self.n_x+x1[mm]
                cnt=np.zeros(k)
                for i,j in zip(idx,jdx):
                    M[i,j]+=1; cnt[i]+=1
                cnt[cnt==0]=1
                M=M/ cnt[:,None]
        self.M=M
        v=self.v.reshape(k)
        r=(self.p_shot*self.p_goal).reshape(k)
        p=(1.0-self.p_shot).reshape(k)
        for _ in range(self.max_iter):
            t=v.copy()
            mv=self.M@v if self.M is not None else np.zeros_like(v)
            v=r+self.gamma*(p*mv)
            if np.max(np.abs(v-t))<self.tol:
                break
        self.v=v.reshape(self.n_y,self.n_x)

    def value_xy(self,x,y):
        xi,yi,m=self._bin_valid(x,y)
        out=np.zeros_like(xi,dtype=float)
        if m.any():
            out[m]=self.v[yi[m],xi[m]]
        return out

    def save(self,path):
        np.savez(path,v=self.v,p_shot=self.p_shot,p_goal=self.p_goal,n_x=self.n_x,n_y=self.n_y,_w=self._w,_h=self._h,gamma=self.gamma)

    @staticmethod
    def load(path):
        z=np.load(path,allow_pickle=True)
        g=XTGrid(int(z["n_x"]),int(z["n_y"]))
        g.v=z["v"]; g.p_shot=z["p_shot"]; g.p_goal=z["p_goal"]
        g._w=float(z["_w"]); g._h=float(z["_h"])
        g.gamma=float(z["gamma"]) if "gamma" in z else 0.97
        return g
