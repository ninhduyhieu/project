
import math, io, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from PIL import Image

# ---------- Load image from local path or URL ----------
def load_image(path_or_url: str) -> Image.Image:
    is_url = path_or_url.lower().startswith(("http://", "https://"))
    if not is_url:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {path_or_url}")
        return Image.open(path_or_url).convert("RGB")
    # URL branch
    try:
        import requests
        r = requests.get(path_or_url, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        import urllib.request
        with urllib.request.urlopen(path_or_url, timeout=20) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")

@dataclass
class Hex:
    q: int; r: int; s: int
    def __post_init__(self): assert self.q + self.r + self.s == 0

def axial_to_pixel(q: int, r: int, size: float = 500.0) -> tuple:
    x = size * (3/2 * q)
    y = size * (math.sqrt(3) * r + (math.sqrt(3)/2) * q)
    return x, y

class CellularNetworkReceivedPower:
    def __init__(self, num_ues=5, manhattan_ratio: float = 0.7,
                 image_path_or_url: Optional[str] = None, rect_len_m: float = 4000.0, rect_wid_m: float = 3000.0,
                 buildings_are_dark: Optional[bool] = None, tiles_x: int = 80, tiles_y: int = 80,
                 occ_thresh: float = 0.35):

        # Core sim
        self.total_size = 10000
        self.center = self.total_size/2
        self.ptx, self.gtx, self.grx = 35, 2, 2           # dBm/dBi
        self.sensitivity = -100                            # dBm
        self.hom, self.fc = 3, 4                           # dB, GHz
        # L3 filter
        self.alpha = 0.2; self.rsrp_avg = {}
        # HO guards
        self.max_bs_range = 3000.0; self.max_candidate_bs = 6

        # CI (UMa)
        self.h_bs, self.h_ut = 25.0, 1.5
        self.scenario = 'UMa'
        self.ple_uma = {'LOS': 2.0, 'NLOS': 2.7}

# Shadow Fading (clip ±3σ)
        self.sf_sigma = {'LOS': 4.6, 'NLOS': 10.0}
        self.sf_decorr = 20.0
        self.sf_cache = {}

        # UE mix
        self.num_ues = int(num_ues)
        n_manhattan = int(round(self.num_ues * manhattan_ratio))
        n_random = self.num_ues - n_manhattan
        self.ue_movement_modes = ['manhattan']*n_manhattan + ['random']*n_random
        np.random.shuffle(self.ue_movement_modes)

        # Rectangle
        self.width, self.height = float(rect_len_m), float(rect_wid_m)
        self.rect_xmin = self.center - self.width/2
        self.rect_xmax = self.center + self.width/2
        self.rect_ymin = self.center - self.height/2
        self.rect_ymax = self.center + self.height/2
        self.boundary = [(self.rect_xmin, self.rect_ymin), (self.rect_xmin, self.rect_ymax),
                         (self.rect_xmax, self.rect_ymax), (self.rect_xmax, self.rect_ymin)]

        # Hex grid cover
        self.scale_factor = 500.0
        self.hexes = self._build_hex_cover()

        # Buildings from image
        if image_path_or_url:
            print(f"Đang đọc ảnh: {image_path_or_url}")
            img = load_image(image_path_or_url)
            self.buildings = self._buildings_from_image(img, tiles_x, tiles_y, occ_thresh, buildings_are_dark)
        else:
            self.buildings = self._fallback_grid_buildings()

        # UE init
        self.ue_positions, self.ue_directions = [], []
        self.ue_serving_bs = [None]*self.num_ues
        self.previous_serving_bs = [None]*self.num_ues

        # Visual & sim params
        self.grid_size = 50
        self.bs_positions = []
        self.bs_colors = ['red','green','blue','cyan','magenta','yellow','black','orange','purple','brown']
        self.ue_colors = ['green','purple','orange','cyan','magenta','yellow','black']
        self.ue_speeds = np.random.uniform(10, 120, self.num_ues)  # km/h
        self.ue_speeds_ms = self.ue_speeds * (1000/3600)
        self.steps, self.time_per_step = 30, 3

        # Matplotlib
        plt.ion()
        self.fig = plt.figure(figsize=(15, 12))
        self.ax = self.fig.add_axes([0.1, 0.1, 0.6, 0.8])
        self.toggle_ax = self.fig.add_axes([0.81, 0.02, 0.08, 0.04])
        self.restart_ax = self.fig.add_axes([0.91, 0.02, 0.08, 0.04])
        from matplotlib.widgets import Button
        self.toggle_button = Button(self.toggle_ax, 'Dừng', color='lightcoral')
        self.restart_button = Button(self.restart_ax, 'Khởi động lại', color='lightgreen')
        self.toggle_button.on_clicked(self.toggle_animation)
        self.restart_button.on_clicked(self.restart_animation)

        self.data_log = {'Bước': []}
        for i in range(self.num_ues):
            for key in ['x','y','huong','BS_ketnoi','ChuyenGiao','prx_hientai','SINR','PacketLoss','speed','handover_count']:
                self.data_log[f'ue{i}_{key}'] = []
        for j in range(len(self.hexes)):
            for i in range(self.num_ues):
                self.data_log.setdefault(f'ue{i}_bs{j}_prx', [])

        self.setup_ues()
        self.setup_plot()

    # -------- hex cover
    def _build_hex_cover(self):
        size = self.scale_factor
        w, h = self.width, self.height
        q_range = range(int(-w/(1.5*size))-1, int(w/(1.5*size))+2)
        r_range = range(int(-h/(size*math.sqrt(3)))-1, int(h/(size*math.sqrt(3)))+2)
        hexes = []
        for q in q_range:
            for r in r_range:
                s = -q - r
                cx, cy = axial_to_pixel(q, r, size)
                cx += self.center; cy += self.center
                buffer = size * math.sqrt(3) / 2
                if (self.rect_xmin - buffer <= cx <= self.rect_xmax + buffer and
                    self.rect_ymin - buffer <= cy <= self.rect_ymax + buffer):
                    hexes.append(Hex(q, r, s))
        return hexes

    # -------- building extraction
    def _buildings_from_image(self, img: Image.Image, tiles_x=80, tiles_y=80, occ_thresh=0.35, buildings_are_dark=None):
        gray = img.convert('L')
        W, H = gray.size
        arr = np.array(gray, dtype=np.uint8)

        # Otsu
        hist = np.bincount(arr.flatten(), minlength=256).astype(np.float64)
        prob = hist / (W*H); omega = np.cumsum(prob); mu = np.cumsum(prob * np.arange(256))
        mu_t = mu[-1]; sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1 - omega) + 1e-12)
        t = int(np.nanargmax(sigma_b2))

        if buildings_are_dark is None:
            buildings_are_dark = arr.mean() < t
        mask = (arr <= t) if buildings_are_dark else (arr >= t)

        bx = max(1, W // tiles_x); by = max(1, H // tiles_y)
        buildings = []; poly_path = MplPath(self.boundary)

        for y0 in range(0, H, by):
            for x0 in range(0, W, bx):
                x1 = min(x0+bx, W); y1 = min(y0+by, H)
                tile = mask[y0:y1, x0:x1]; occ = tile.mean()
                if occ >= occ_thresh:
                    wx1 = self.rect_xmin + (x0 / W) * self.width
                    wx2 = self.rect_xmin + (x1 / W) * self.width
                    wy1 = self.rect_ymin + ((H - y1) / H) * self.height
                    wy2 = self.rect_ymin + ((H - y0) / H) * self.height
                    cx = 0.5*(wx1+wx2); cy = 0.5*(wy1+wy2)
                    if poly_path.contains_point((cx, cy)):
                        buildings.append({"x1":wx1,"y1":wy1,"x2":wx2,"y2":wy2,
                                          "height": float(np.random.randint(15,60)),
                                          "occ": float(occ)})
        area_rect = self.width*self.height
        for b in buildings:
            area_b = (b["x2"]-b["x1"])*(b["y2"]-b["y1"])
            b["big"] = (area_b >= 0.015*area_rect) or (b["occ"] >= 0.65)
        print(f"Trích xuất {len(buildings)} toà nhà (dark={buildings_are_dark}).")
        return buildings

    def _fallback_grid_buildings(self, rows=12, cols=12, building_w=400, building_h=400, street_spacing=60, group_size=4, group_gap=240):
        buildings = []; poly_path = MplPath(self.boundary)
        total_w = cols*building_w + (cols-1)*street_spacing + ((cols-1)//group_size)*group_gap
        total_h = rows*building_h + (rows-1)*street_spacing + ((rows-1)//group_size)*group_gap
        offset_x = ((self.rect_xmax-self.rect_xmin)-total_w)/2 + self.rect_xmin
        offset_y = ((self.rect_ymax-self.rect_ymin)-total_h)/2 + self.rect_ymin
        for i in range(rows):
            for j in range(cols):
                ex = (j//group_size)*group_gap; ey = (i//group_size)*group_gap
                x1 = j*(building_w+street_spacing)+ex+offset_x
                y1 = i*(building_h+street_spacing)+ey+offset_y
                x2 = x1+building_w; y2 = y1+building_h
                cx = (x1+x2)/2; cy = (y1+y2)/2
                if poly_path.contains_point((cx,cy)):
                    buildings.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"height":np.random.randint(15,40),"big":False})
        return buildings

    # -------- geometry & LOS
    def _rect_edges(self, b):
        return [((b["x1"], b["y1"]), (b["x2"], b["y1"])),
                ((b["x2"], b["y1"]), (b["x2"], b["y2"])),
                ((b["x2"], b["y2"]), (b["x1"], b["y2"])),
                ((b["x1"], b["y2"]), (b["x1"], b["y1"]))]

    def line_segment_intersection(self, p1, p2, q1, q2):
        x1,y1=p1; x2,y2=p2; x3,y3=q1; x4,y4=q2
        denom=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        if abs(denom)<1e-10: return None
        t=((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/denom
        u=-((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/denom
        if 0<=t<=1 and 0<=u<=1: return (x1+t*(x2-x1), y1+t*(y2-y1))
        return None

    def _blocked_by_buildings(self, p1, p2, eps=1e-6):
        for b in self.buildings:
            for e1,e2 in self._rect_edges(b):
                inter=self.line_segment_intersection(p1,p2,e1,e2)
                if inter is not None:
                    if (abs(inter[0]-p1[0])>eps or abs(inter[1]-p1[1])>eps) and \
                       (abs(inter[0]-p2[0])>eps or abs(inter[1]-p2[1])>eps):
                        return True
        return False

    def is_los(self, ue_x, ue_y, bs_x, bs_y):
        return not self._blocked_by_buildings((ue_x, ue_y), (bs_x, bs_y))

    # -------- channel
    def calculate_distance(self, ue_x, ue_y, bs_x, bs_y):
        d2 = np.hypot(ue_x-bs_x, ue_y-bs_y); dz = self.h_bs - self.h_ut
        return max(float(np.sqrt(d2**2 + dz**2)), 1.0)

    def calculate_path_loss(self, ue_x, ue_y, bs_x, bs_y):
        d3 = self.calculate_distance(ue_x, ue_y, bs_x, bs_y)
        los = self.is_los(ue_x, ue_y, bs_x, bs_y)
        fspl_1m = 32.4 + 20*np.log10(self.fc)  # f in GHz
        n = self.ple_uma['LOS'] if los else self.ple_uma['NLOS']
        pl = fspl_1m + 10*n*np.log10(d3)
        return pl, los

    def _shadow_fading(self, ue_idx, bs_idx, los, ue_pos):
        sigma = self.sf_sigma['LOS'] if los else self.sf_sigma['NLOS']
        key = (ue_idx, bs_idx); state = self.sf_cache.get(key)
        if state is None:
            val=float(np.random.normal(0.0, sigma)); val=float(np.clip(val, -3*sigma, 3*sigma))
            self.sf_cache[key]={'x':ue_pos[0],'y':ue_pos[1],'val':val,'los':los}; return val
        oldx,oldy=state['x'],state['y']; oldval=state['val']
        delta=float(np.hypot(ue_pos[0]-oldx, ue_pos[1]-oldy))
        rho=0.0 if self.sf_decorr<=0 else float(np.exp(-delta/self.sf_decorr))
        innov=float(np.random.normal(0.0, sigma))
        newval=rho*oldval + (np.sqrt(max(0.0,1.0-rho**2))*innov)
        newval=float(np.clip(newval, -3*sigma, 3*sigma))
        self.sf_cache[key]={'x':ue_pos[0],'y':ue_pos[1],'val':newval,'los':los}; return newval

    def calculate_received_power(self, path_loss):
        return self.ptx + self.gtx + self.grx - path_loss

    # -------- serving cell selection
    
    def _log_per_bs_prx(self, ue_idx):
        ue_x, ue_y = self.ue_positions[ue_idx]
        for j, (bs_x, bs_y) in enumerate(self.bs_positions):
            pl_base, los = self.calculate_path_loss(ue_x, ue_y, bs_x, bs_y)
            sf_db = self._shadow_fading(ue_idx, j, los, (ue_x, ue_y))
            prx = self.calculate_received_power(pl_base + sf_db)
            self.data_log.setdefault(f'ue{ue_idx}_bs{j}_prx', []).append(prx)
    def get_serving_bs(self, ue_x, ue_y, ue_idx):
            dist_list=[]
            for i,(bs_x, bs_y) in enumerate(self.bs_positions):
                if self.max_bs_range is not None:
                    d2=float(np.hypot(ue_x-bs_x, ue_y-bs_y))
                    if d2>self.max_bs_range: continue
                d3=self.calculate_distance(ue_x, ue_y, bs_x, bs_y)
                dist_list.append((i,d3))
            dist_list.sort(key=lambda x:x[1])
            candidate_indices=[i for i,_ in dist_list[:self.max_candidate_bs]]

            received_powers=[]
            for i in candidate_indices:
                bs_x, bs_y = self.bs_positions[i]
                d3 = self.calculate_distance(ue_x, ue_y, bs_x, bs_y)
                pl, los = self.calculate_path_loss(ue_x, ue_y, bs_x, bs_y)
                sf_db = self._shadow_fading(ue_idx, i, los, (ue_x, ue_y))
                rsrp_inst = self.calculate_received_power(pl + sf_db)

                key=(ue_idx,i)
                prev=self.rsrp_avg.get(key, rsrp_inst)
                rsrp_avg=(1-self.alpha)*prev + self.alpha*rsrp_inst
                self.rsrp_avg[key]=rsrp_avg

                if rsrp_inst >= self.sensitivity:
                    received_powers.append((i, rsrp_avg, d3, pl+sf_db))

            if not received_powers:
                self.ue_serving_bs[ue_idx]=None; return None, None, None

            received_powers.sort(key=lambda x:x[1], reverse=True)
            if self.ue_serving_bs[ue_idx] is None:
                best_bs,best_prx,best_d,_=received_powers[0]
                self.ue_serving_bs[ue_idx]=best_bs; return best_bs,best_prx,best_d

            current=next((x for x in received_powers if x[0]==self.ue_serving_bs[ue_idx]), None)
            if current is None:
                best_bs,best_prx,best_d,_=received_powers[0]
                self.ue_serving_bs[ue_idx]=best_bs; return best_bs,best_prx,best_d

            prx_curr=current[1]
            dynamic_hom = self.hom + (self.ue_speeds[ue_idx]/100)*3
            for bs_idx, bs_prx, bs_distance, _ in received_powers:
                if bs_idx!=self.ue_serving_bs[ue_idx] and bs_prx>prx_curr+dynamic_hom:
                    self.ue_serving_bs[ue_idx]=bs_idx; return bs_idx, bs_prx, bs_distance
            return self.ue_serving_bs[ue_idx], current[1], current[2]

    def calculate_sinr_and_packet_loss(self, ue_idx, serving_bs_idx, prx_dbm):
            N_dbm=-100; N_mw=10**(N_dbm/10); interf=0.0
            for i,(bs_x,bs_y) in enumerate(self.bs_positions):
                if i==serving_bs_idx: continue
                if self.max_bs_range is not None:
                    d2=float(np.hypot(self.ue_positions[ue_idx][0]-bs_x, self.ue_positions[ue_idx][1]-bs_y))
                    if d2>self.max_bs_range: continue
                pl, los = self.calculate_path_loss(self.ue_positions[ue_idx][0], self.ue_positions[ue_idx][1], bs_x, bs_y)
                sf_db = self._shadow_fading(ue_idx, i, los, (self.ue_positions[ue_idx][0], self.ue_positions[ue_idx][1]))
                prx_i = self.calculate_received_power(pl + sf_db)
                if prx_i >= self.sensitivity: interf += 10**(prx_i/10)
            prx_mw=10**(prx_dbm/10)
            sinr_lin=prx_mw/(interf+N_mw)
            sinr_db=10*np.log10(sinr_lin) if sinr_lin>0 else -np.inf
            k=0.1; ploss=np.exp(-k*sinr_db) if sinr_db>0 else 1.0
            return 10*np.log10(sinr_lin) if sinr_lin>0 else -np.inf, min(max(ploss,0),1)

        # -------- plot & animation
    def setup_ues(self):
            self.ue_positions.clear(); self.ue_directions.clear()
            for _ in range(self.num_ues):
                x=np.random.uniform(self.rect_xmin, self.rect_xmax)
                y=np.random.uniform(self.rect_ymin, self.rect_ymax)
                self.ue_positions.append([x,y])
                self.ue_directions.append(np.random.randint(0,4))

    def _draw_hex(self, cx, cy, size=None, edgecolor='blue'):
            if size is None: size=self.scale_factor
            angles=[60*i for i in range(6)]
            xs=[cx+size*math.cos(math.radians(a)) for a in angles]+[cx+size*math.cos(0)]
            ys=[cy+size*math.sin(math.radians(a)) for a in angles]+[cy+size*math.sin(0)]
            self.ax.plot(xs,ys,color=edgecolor,linewidth=1)

    def setup_plot(self):
            # grid
            step=int(self.total_size/self.grid_size)
            for x in range(0,self.total_size+1,step):
                self.ax.plot([x,x],[0,self.total_size],'gray',linestyle=':',alpha=0.4,linewidth=0.8)
                self.ax.plot([0,self.total_size],[x,x],'gray',linestyle=':',alpha=0.4,linewidth=0.8)
            # rectangle
            self.ax.add_patch(Rectangle((self.rect_xmin,self.rect_ymin), self.width,self.height,
                                        linewidth=2, edgecolor='red', facecolor='none', linestyle='--', zorder=5))
            # buildings (fill as light blue patches for NLOS masks)
            for b in self.buildings:
                w = b["x2"] - b["x1"]
                h = b["y2"] - b["y1"]
                # Light ocean blue fill, semi-transparent, no edges
                self.ax.add_patch(Rectangle((b["x1"], b["y1"]), w, h,
                                           facecolor="#7ec8e3", edgecolor='none', alpha=0.35, zorder=2))

            # hex & bs
            self.bs_positions.clear()
            for idx,hex_obj in enumerate(self.hexes):
                x,y=axial_to_pixel(hex_obj.q,hex_obj.r,self.scale_factor)
                cx,cy=x+self.center,y+self.center
                self._draw_hex(cx,cy)
                self.bs_positions.append((cx,cy))
                self.ax.plot(cx,cy,marker='^',color=self.bs_colors[idx%len(self.bs_colors)],markersize=8,label=f'BS{idx}')
            # UE visuals
            self.ue_points=[]; self.ue_texts=[]; self.ue_lines=[]
            for i in range(self.num_ues):
                color=self.ue_colors[i%len(self.ue_colors)]
                p,=self.ax.plot([],[],'o',color=color,markersize=5,label=f'UE{i}')
                t=self.ax.text(0,0,'',fontsize=8); l,=self.ax.plot([],[],'-',color=color,linewidth=1,alpha=0.7)
                self.ue_points.append(p); self.ue_texts.append(t); self.ue_lines.append(l)
            self.time_text=self.ax.text(self.rect_xmin+20,self.rect_ymax+20,'',fontsize=10)
            self.ax.set_xlim(0,self.total_size); self.ax.set_ylim(0,self.total_size)
            self.ax.set_xlabel('Khoảng cách (m)'); self.ax.set_ylabel('Khoảng cách (m)')
            self.ax.set_title(f'Ảnh → Building (NLOS mask: xanh biển nhạt) → LOS/NLOS | Khu vực: {int(self.width)} x {int(self.height)} m')
            self.ax.legend(bbox_to_anchor=(1.05,1),loc='upper left'); plt.draw()

    def toggle_animation(self, event):
            self.animation_running = not getattr(self, 'animation_running', False)
            if self.animation_running:
                self.toggle_button.label.set_text('Dừng'); self.toggle_button.color='lightcoral'; self.run_animation()
            else:
                self.toggle_button.label.set_text('Tiếp tục'); self.toggle_button.color='lightblue'
            self.toggle_button.ax.figure.canvas.draw()

    def restart_animation(self, event):
            self.animation_running=True; self.current_frame=0
            self.ue_serving_bs=[None]*self.num_ues; self.previous_serving_bs=[None]*self.num_ues
            self.rsrp_avg.clear(); self.data_log={'Bước':[]}; self.setup_ues()
            self.toggle_button.label.set_text('Dừng'); self.toggle_button.color='lightcoral'
            self.toggle_button.ax.figure.canvas.draw(); self.run_animation()

    def update(self, frame):
            if not getattr(self,'animation_running',True): return
            self.current_frame=frame; self.data_log['Bước'].append(frame)
            for ue_idx in range(self.num_ues):
                d = self.ue_speeds_ms[ue_idx]*self.time_per_step
                m = self.ue_movement_modes[ue_idx]
                x,y = self.ue_positions[ue_idx]
                if m=='manhattan':
                    axis=np.random.choice([0,1]); s=np.random.choice([-1,1])
                    if axis==0: x+=s*d
                    else: y+=s*d
                else:
                    dir = self.ue_directions[ue_idx]
                    rv=np.random.random()
                    if rv<0.5: pass
                    elif rv<0.75: dir=(dir+1)%4
                    else: dir=(dir-1+4)%4
                    if abs(dir-self.ue_directions[ue_idx])==2: dir=self.ue_directions[ue_idx]
                    self.ue_directions[ue_idx]=dir
                    if dir==0: x+=d
                    elif dir==1: y+=d
                    elif dir==2: x-=d
                    else: y-=d
                # keep in rect
                if self.rect_xmin<=x<=self.rect_xmax and self.rect_ymin<=y<=self.rect_ymax:
                    self.ue_positions[ue_idx]=[x,y]
                else:
                    self.ue_directions[ue_idx]=(self.ue_directions[ue_idx]+1)%4
                # draw
                self.ue_points[ue_idx].set_data([self.ue_positions[ue_idx][0]],[self.ue_positions[ue_idx][1]])
                bs, prx_avg, dist = self.get_serving_bs(self.ue_positions[ue_idx][0], self.ue_positions[ue_idx][1], ue_idx)
                dyn_hom = self.hom + (self.ue_speeds[ue_idx]/100)*3
                handover = int(self.previous_serving_bs[ue_idx] is not None and bs != self.previous_serving_bs[ue_idx])
                if handover: self.handover_count[ue_idx]+=1
                if bs is not None:
                    pl_base, los = self.calculate_path_loss(self.ue_positions[ue_idx][0], self.ue_positions[ue_idx][1], *self.bs_positions[bs])
                    sf_db = self._shadow_fading(ue_idx, bs, los, (self.ue_positions[ue_idx][0], self.ue_positions[ue_idx][1]))
                    prx_inst = self.calculate_received_power(pl_base + sf_db)
                    sinr, pkt = self.calculate_sinr_and_packet_loss(ue_idx, bs, prx_inst)
                    self.ue_texts[ue_idx].set_position((self.ue_positions[ue_idx][0]+30, self.ue_positions[ue_idx][1]+30))
                    self.ue_texts[ue_idx].set_text(f'UE{ue_idx}\\nBS:{bs}\\nPrx(avg):{prx_avg:.1f} dBm\\nPrx(inst):{prx_inst:.1f} dBm\\nSINR:{sinr:.1f} dB\\nPL:{pkt*100:.1f}%\\nDist:{dist:.0f} m')
                    self.ue_lines[ue_idx].set_data([self.ue_positions[ue_idx][0], self.bs_positions[bs][0]],[self.ue_positions[ue_idx][1], self.bs_positions[bs][1]])
                else:
                    self.ue_texts[ue_idx].set_text(''); self.ue_lines[ue_idx].set_data([],[])
                # logs
                self.data_log[f'ue{ue_idx}_x'].append(self.ue_positions[ue_idx][0])
                self.data_log[f'ue{ue_idx}_y'].append(self.ue_positions[ue_idx][1])
                self.data_log[f'ue{ue_idx}_huong'].append(self.ue_directions[ue_idx])
                self.data_log[f'ue{ue_idx}_BS_ketnoi'].append(bs)
                self.data_log[f'ue{ue_idx}_prx_hientai'].append(prx_inst if bs is not None else None)
                self.data_log[f'ue{ue_idx}_speed'].append(self.ue_speeds[ue_idx])
                self.data_log[f'ue{ue_idx}_ChuyenGiao'].append(handover)
                self.previous_serving_bs[ue_idx]=bs
            self.time_text.set_text(f'Bước: {frame}')
            self.fig.canvas.draw(); self.fig.canvas.flush_events()

        
            # Log per-BS PRX each frame
            for ue_idx in range(self.num_ues):
                self._log_per_bs_prx(ue_idx)
            # Save CSV on last frame
            if frame == self.steps:
                try:
                    self.save_data_to_csv()
                except Exception as e:
                    print('CSV save failed:', e)

    def save_data_to_csv(self):
        # Pad columns to equal length
        max_len = max((len(v) for v in self.data_log.values()), default=0)
        for k in self.data_log:
            while len(self.data_log[k]) < max_len:
                self.data_log[k].append(None)
        df = pd.DataFrame(self.data_log)

        # Round numeric cols of interest
        for col in df.columns:
            if any(tag in col for tag in ['prx', '_x', '_y']):
                df[col] = pd.to_numeric(df[col], errors='coerce').round(1)

        # Order columns
        cols = ['Bước']
        for i in range(self.num_ues):
            cols += [f'ue{i}_x', f'ue{i}_y', f'ue{i}_huong', f'ue{i}_BS_ketnoi',
                     f'ue{i}_prx_hientai', f'ue{i}_ChuyenGiao',
                      f'ue{i}_speed',
                     ]
            for j in range(len(self.bs_positions)):
                cols.append(f'ue{i}_bs{j}_prx')
        cols = [c for c in cols if c in df.columns]
        df = df.reindex(columns=cols)

        filename = f"du_lieu_cong_suat_nhan_ues{self.num_ues}_vantoc{float(np.mean(self.ue_speeds)):.1f}_thoigianbuoc{self.time_per_step}_ptx{self.ptx}_buoc{self.steps}_hom{self.hom}.csv"
        df.to_csv(filename, index=False, sep=';', decimal=',', encoding='utf-8-sig', float_format='%.2f')
        print(f"Đã lưu CSV: {filename}")

    def run_animation(self):
            self.animation_running=True; self.current_frame=0
            for frame in range(0, self.steps+1):
                if not self.animation_running: break
                self.update(frame); plt.pause(0.1)


            try:
                self.save_data_to_csv()
            except Exception as e:
                print('CSV save failed:', e)
    # --------------- CLI ---------------
if __name__ == "__main__":
    path_or_url = input("Dán ĐƯỜNG DẪN ẢNH (local path hoặc URL): ").strip() or None
    try:
        L = float(input("Nhập chiều DÀI HCN (m): ")); W = float(input("Nhập chiều RỘNG HCN (m): "))
    except: L, W = 4000.0, 3000.0
    try: num_ues = int(input("Nhập số UE: "))
    except: num_ues = 5
    try:
        r = input("Tỷ lệ Manhattan (0..1, mặc định 0.7): ").strip()
        ratio = float(r) if r else 0.7
        ratio = min(max(ratio, 0.0), 1.0)
    except: ratio = 0.7

    opt = input("Buildings trong ảnh là MÀU TỐI? (y/n, Enter=auto): ").strip().lower()
    if opt == 'y': dark = True
    elif opt == 'n': dark = False
    else: dark = None

    sim = CellularNetworkReceivedPower(num_ues=num_ues, manhattan_ratio=ratio,
                                       image_path_or_url=path_or_url, rect_len_m=L, rect_wid_m=W,
                                       buildings_are_dark=dark)
    print(f"Số hex cells: {len(sim.hexes)} | Khu vực: {int(L)} x {int(W)} m")
    sim.run_animation(); plt.ioff(); plt.show()