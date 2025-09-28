
import math, io, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image

try:
    from numba import njit
except Exception:
    njit = None

def _clean_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    p = os.path.expanduser(os.path.expandvars(p))
    return p

def load_image(path_or_url: str) -> Image.Image:
    path_or_url = _clean_path(path_or_url)
    is_url = path_or_url.lower().startswith(("http://", "https://"))
    if not is_url:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {path_or_url}")
        return Image.open(path_or_url).convert("RGB")
    try:
        import requests
        r = requests.get(path_or_url, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
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

if njit is not None:
    @njit(cache=True, fastmath=True)
    def _frac_black_on_segment_rows(mask, col1, row1, col2, row2, n_samples):
        cnt = 0
        for i in range(n_samples):
            t = i/(n_samples-1) if n_samples>1 else 0.0
            c = int(col1 + t*(col2-col1))
            r = int(row1 + t*(row2-row1))
            if r<0 or r>=mask.shape[0] or c<0 or c>=mask.shape[1]:
                continue
            cnt += 1 if mask[r, c] != 0 else 0
        return cnt / max(n_samples, 1)
else:
    _frac_black_on_segment_rows = None

class CellularNetworkReceivedPower:
    def __init__(self, num_ues=5, manhattan_ratio: float = 0.7,
                 image_path_or_url: Optional[str] = None, rect_len_m: float = 4000.0, rect_wid_m: float = 3000.0,
                 sample_step_m: float = 10.0, black_ratio_los: float = 0.50,
                 black_threshold: int = 64, draw_hex: bool = True,
                 show_link_lines: bool = True, fast_mode: bool = True,
                 # --- NEW parameters for "white area" logic ---
                 white_win_radius_m: float = 20.0,    # bán kính cửa sổ tính mật độ TRẮNG (m)
                 white_density_thresh: float = 0.55,  # nếu mật độ trắng >= ngưỡng -> xem như "vùng nhiều mảng trắng"
                 white_slow_factor: float = 0.4,      # hệ số giảm tốc (0.4 => còn 40% tốc độ gốc)
                 stop_after_entries: int = 3,         # cứ 3 lần vào vùng trắng thì dừng...
                 stop_duration_steps: int = 10        # ...trong 10 timestep
                 ):

        self.image_origin = 'upper'  # giữ đúng chiều ảnh input

        # Core sim
        self.total_size = 10000
        self.center = self.total_size/2
        self.ptx, self.gtx, self.grx = 35, 2, 2
        self.sensitivity = -100
        self.hom, self.fc = 3, 4

        self.alpha = 0.2; self.rsrp_avg = {}
        self.max_bs_range = 3000.0; self.max_candidate_bs = 6

        self.h_bs, self.h_ut = 25.0, 1.5
        self.ple_uma = {'LOS': 2.0, 'NLOS': 2.7}

        self.sf_sigma = {'LOS': 4.6, 'NLOS': 10.0}
        self.sf_decorr = 20.0
        self.sf_cache = {}

        self.num_ues = int(num_ues)
        n_manhattan = int(round(self.num_ues * manhattan_ratio))
        n_random = self.num_ues - n_manhattan
        self.ue_movement_modes = ['manhattan']*n_manhattan + ['random']*n_random
        np.random.shuffle(self.ue_movement_modes)

        self.width, self.height = float(rect_len_m), float(rect_wid_m)
        self.rect_xmin = self.center - self.width/2
        self.rect_xmax = self.center + self.width/2
        self.rect_ymin = self.center - self.height/2
        self.rect_ymax = self.center + self.height/2

        self.scale_factor = 500.0
        self.hexes = self._build_hex_cover()

        if not image_path_or_url:
            raise ValueError("Cần CUNG CẤP ẢNH nền trắng, vùng MÀU ĐEN là vật cản/đường.")
        print(f"Đang đọc ảnh (mask màu ĐEN): {image_path_or_url}")
        self.input_image = load_image(image_path_or_url)
        self.sample_step_m = float(sample_step_m)
        self.black_ratio_los = float(black_ratio_los)
        self.black_threshold = int(np.clip(black_threshold, 0, 255))
        self._prepare_black_mask(self.input_image)

        # --- NEW: white area detection configs ---
        self.white_win_radius_m = float(max(1.0, white_win_radius_m))
        self.white_density_thresh = float(np.clip(white_density_thresh, 0.0, 1.0))
        self.white_slow_factor = float(np.clip(white_slow_factor, 0.01, 1.0))
        self.stop_after_entries = max(1, int(stop_after_entries))
        self.stop_duration_steps = max(1, int(stop_duration_steps))

        self.grid_size = 50
        self.bs_colors = ['red','green','blue','cyan','magenta','yellow','black','orange','purple','brown']
        self.ue_colors = ['green','purple','orange','cyan','magenta','yellow','black']
        self.ue_speeds = np.random.uniform(20, 120, self.num_ues)  # km/h
        self.ue_speeds_ms = self.ue_speeds * (1000/3600)
        self.steps, self.time_per_step = (150 if fast_mode else 100), 3
        self.pause_s = (0.03 if fast_mode else 0.08)
        self.draw_hex = bool(draw_hex)
        self.show_link_lines = bool(show_link_lines)

        self.ue_positions, self.ue_directions = [], []
        self.ue_serving_bs = [None]*self.num_ues
        self.previous_serving_bs = [None]*self.num_ues

        # --- NEW: state for white-area logic ---
        self.in_white_zone = [False]*self.num_ues
        self.white_entries = [0]*self.num_ues
        self.stop_counters = [0]*self.num_ues  # nếu >0 thì UE đứng yên trong số bước còn lại

        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_axes([0.08, 0.1, 0.84, 0.82])  # rộng vì không có legend
        self.toggle_ax = self.fig.add_axes([0.08, 0.02, 0.12, 0.05])
        self.restart_ax = self.fig.add_axes([0.22, 0.02, 0.14, 0.05])
        self.toggle_button = Button(self.toggle_ax, 'Dừng', color='lightcoral')
        self.restart_button = Button(self.restart_ax, 'Khởi động lại', color='lightgreen')
        self.toggle_button.on_clicked(self.toggle_animation)
        self.restart_button.on_clicked(self.restart_animation)

        self.data_log = {'Bước': []}
        for i in range(self.num_ues):
            for key in ['x','y','huong','BS_ketnoi','prx_hientai','SINR','PacketLoss','speed','speed_eff',
                        'white_zone','white_entries','stop_remain']:
                self.data_log[f'ue{i}_{key}'] = []
        self.setup_ues()
        self.setup_plot()

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

    def _prepare_black_mask(self, img: Image.Image):
        arr = np.asarray(img).astype(np.uint8)
        gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.uint8)
        self.black_mask = (gray <= self.black_threshold).astype(np.uint8)
        self.white_mask = (1 - self.black_mask).astype(np.uint8)  # TRẮNG = không đen
        self.img_W, self.img_H = img.size

    def _world_to_img_cols_rows(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = (x - self.rect_xmin) / max(self.width, 1e-9)
        v = (y - self.rect_ymin) / max(self.height, 1e-9)
        cols = np.clip((u * self.img_W).astype(np.int64), 0, self.img_W - 1)
        rows = np.clip(((1.0 - v) * self.img_H).astype(np.int64), 0, self.img_H - 1)  # origin=UPPER
        return cols, rows

    def _black_fraction_on_segment(self, p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
        x1,y1 = p1; x2,y2 = p2
        seg_len = float(np.hypot(x2-x1, y2-y1))
        if seg_len < 1e-6:
            cols, rows = self._world_to_img_cols_rows(np.array([x1]), np.array([y1]))
            return float(self.black_mask[rows[0], cols[0]])
        n_samples = max(2, int(math.ceil(seg_len / max(self.sample_step_m, 1.0))))
        ts = np.linspace(0.0, 1.0, n_samples)
        xs = x1 + ts*(x2-x1)
        ys = y1 + ts*(y2-y1)
        valid = (xs>=self.rect_xmin) & (xs<=self.rect_xmax) & (ys>=self.rect_ymin) & (ys<=self.rect_ymax)
        if not np.any(valid): return 0.0
        xs = xs[valid]; ys = ys[valid]
        cols, rows = self._world_to_img_cols_rows(xs, ys)
        if _frac_black_on_segment_rows is not None:
            return float(_frac_black_on_segment_rows(self.black_mask, cols[0], rows[0], cols[-1], rows[-1], len(cols)))
        else:
            return float(self.black_mask[rows, cols].mean())

    def is_los(self, ue_x, ue_y, bs_x, bs_y):
        frac_black = self._black_fraction_on_segment((ue_x, ue_y), (bs_x, bs_y))
        return frac_black >= self.black_ratio_los

    def _is_in_dense_white(self, x: float, y: float) -> bool:
        """Kiểm tra quanh điểm (x,y) có 'mật độ TRẮNG' đủ lớn không."""
        # Chuyển (m) -> (px)
        cols, rows = self._world_to_img_cols_rows(np.array([x]), np.array([y]))
        c = int(cols[0]); r = int(rows[0])
        # độ phân giải mét/px (xấp xỉ theo trục x)
        m_per_px_x = self.width / max(self.img_W, 1)
        win_half_px = max(1, int(round(self.white_win_radius_m / max(m_per_px_x, 1e-6))))
        r0 = max(0, r - win_half_px); r1 = min(self.img_H, r + win_half_px + 1)
        c0 = max(0, c - win_half_px); c1 = min(self.img_W, c + win_half_px + 1)
        region = self.white_mask[r0:r1, c0:c1]
        if region.size == 0: return False
        white_density = float(region.mean())
        return white_density >= self.white_density_thresh

    def calculate_distance(self, ue_x, ue_y, bs_x, bs_y):
        d2 = np.hypot(ue_x-bs_x, ue_y-bs_y); dz = self.h_bs - self.h_ut
        return max(float(np.sqrt(d2**2 + dz**2)), 1.0)

    def calculate_path_loss(self, ue_x, ue_y, bs_x, bs_y):
        d3 = self.calculate_distance(ue_x, ue_y, bs_x, bs_y)
        los = self.is_los(ue_x, ue_y, bs_x, bs_y)
        fspl_1m = 32.4 + 20*np.log10(self.fc)
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
            pl, los = self.calculate_path_loss(ue_x, ue_y, bs_x, bs_y)
            sf_db = self._shadow_fading(ue_idx, i, los, (ue_x, ue_y))
            rsrp_inst = self.calculate_received_power(pl + sf_db)
            key=(ue_idx,i)
            prev=self.rsrp_avg.get(key, rsrp_inst)
            rsrp_avg=(1-self.alpha)*prev + self.alpha*rsrp_inst
            self.rsrp_avg[key]=rsrp_avg
            if rsrp_inst >= self.sensitivity:
                received_powers.append((i, rsrp_avg, self.calculate_distance(ue_x, ue_y, bs_x, bs_y), pl+sf_db))

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
        step=int(self.total_size/self.grid_size)
        for x in range(0,self.total_size+1,step):
            self.ax.plot([x,x],[0,self.total_size],'gray',linestyle=':',alpha=0.25,linewidth=0.8,zorder=1)
            self.ax.plot([0,self.total_size],[x,x],'gray',linestyle=':',alpha=0.25,linewidth=0.8,zorder=1)

        self.ax.imshow(self.input_image, extent=[self.rect_xmin, self.rect_xmax, self.rect_ymin, self.rect_ymax],
                       origin=self.image_origin, zorder=0, alpha=1.0)

        self.ax.add_patch(Rectangle((self.rect_xmin,self.rect_ymin), self.width,self.height,
                                    linewidth=2, edgecolor='red', facecolor='none', linestyle='--', zorder=5))

        self.bs_positions=[]
        for idx,hex_obj in enumerate(self.hexes):
            x,y=axial_to_pixel(hex_obj.q,hex_obj.r,self.scale_factor)
            cx,cy=x+self.center,y+self.center
            if self.draw_hex: self._draw_hex(cx,cy)
            self.bs_positions.append((cx,cy))
            self.ax.plot(cx,cy,marker='^',color=self.bs_colors[idx%len(self.bs_colors)],markersize=5,zorder=6)

        # UE visuals + link lines (restored)
        self.ue_points=[]; self.ue_lines=[]
        for i in range(self.num_ues):
            color=self.ue_colors[i%len(self.ue_colors)]
            p,=self.ax.plot([],[],'o',color=color,markersize=3,zorder=7)
            self.ue_points.append(p)
            l,=self.ax.plot([],[],'-',color=color,linewidth=0.8,alpha=(0.6 if self.show_link_lines else 0.0),zorder=6)
            self.ue_lines.append(l)

        self.time_text=self.ax.text(self.rect_xmin+20,self.rect_ymax+20,'',fontsize=10,zorder=8)
        self.ax.set_xlim(0,self.total_size); self.ax.set_ylim(0,self.total_size)
        self.ax.set_xlabel('Khoảng cách (m)'); self.ax.set_ylabel('Khoảng cách (m)')
        self.ax.set_title(
            'LOS nếu đi qua MÀU ĐEN > 50% | origin=UPPER | Khu vực: '
            f'{int(self.width)} x {int(self.height)} m | Vùng TRẮNG: chậm + 3 lần -> dừng 10 bước'
        )
        plt.draw()

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
        self.rsrp_avg.clear(); self.data_log={'Bước':[]}
        self.in_white_zone=[False]*self.num_ues; self.white_entries=[0]*self.num_ues; self.stop_counters=[0]*self.num_ues
        self.setup_ues()
        # re-create data_log columns
        for i in range(self.num_ues):
            for key in ['x','y','huong','BS_ketnoi','prx_hientai','SINR','PacketLoss','speed','speed_eff',
                        'white_zone','white_entries','stop_remain']:
                self.data_log[f'ue{i}_{key}'] = []
        self.toggle_button.label.set_text('Dừng'); self.toggle_button.color='lightcoral'
        self.toggle_button.ax.figure.canvas.draw(); self.run_animation()

    def update(self, frame):
        if not getattr(self,'animation_running',True): return
        self.current_frame=frame; self.data_log['Bước'].append(frame)
        for ue_idx in range(self.num_ues):
            base_d = self.ue_speeds_ms[ue_idx]*self.time_per_step
            x,y = self.ue_positions[ue_idx]

            # --- NEW: detect white zone at current position (pre-move) ---
            now_white = self._is_in_dense_white(x, y)
            # Edge detection for "entry"
            if now_white and (not self.in_white_zone[ue_idx]):
                self.white_entries[ue_idx] += 1
                # Every Nth entry -> stop for configured steps
                if (self.white_entries[ue_idx] % self.stop_after_entries) == 0:
                    self.stop_counters[ue_idx] = self.stop_duration_steps

            self.in_white_zone[ue_idx] = now_white

            # --- NEW: apply stop or slow ---
            if self.stop_counters[ue_idx] > 0:
                eff_factor = 0.0
                self.stop_counters[ue_idx] -= 1
            else:
                eff_factor = (self.white_slow_factor if now_white else 1.0)

            d = base_d * eff_factor

            # Movement model (unchanged)
            m = self.ue_movement_modes[ue_idx]
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

            if self.rect_xmin<=x<=self.rect_xmax and self.rect_ymin<=y<=self.rect_ymax:
                self.ue_positions[ue_idx]=[x,y]
            else:
                self.ue_directions[ue_idx]=(self.ue_directions[ue_idx]+1)%4
            self.ue_points[ue_idx].set_data([self.ue_positions[ue_idx][0]],[self.ue_positions[ue_idx][1]])

            bs, prx_avg, dist = self.get_serving_bs(self.ue_positions[ue_idx][0], self.ue_positions[ue_idx][1], ue_idx)

            # Update link line
            if bs is not None and self.show_link_lines:
                self.ue_lines[ue_idx].set_data([self.ue_positions[ue_idx][0], self.bs_positions[bs][0]],
                                               [self.ue_positions[ue_idx][1], self.bs_positions[bs][1]])
            else:
                self.ue_lines[ue_idx].set_data([],[])

            # Logging (CSV only)
            if bs is not None:
                pl_base, los = self.calculate_path_loss(self.ue_positions[ue_idx][0], self.ue_positions[ue_idx][1], *self.bs_positions[bs])
                sf_db = self._shadow_fading(ue_idx, bs, los, (self.ue_positions[ue_idx][0], self.ue_positions[ue_idx][1]))
                prx_inst = self.calculate_received_power(pl_base + sf_db)
                sinr, pkt = self.calculate_sinr_and_packet_loss(ue_idx, bs, prx_inst)
            else:
                prx_inst=None; sinr=None; pkt=None

            self.data_log[f'ue{ue_idx}_x'].append(self.ue_positions[ue_idx][0])
            self.data_log[f'ue{ue_idx}_y'].append(self.ue_positions[ue_idx][1])
            self.data_log[f'ue{ue_idx}_huong'].append(self.ue_directions[ue_idx])
            self.data_log[f'ue{ue_idx}_BS_ketnoi'].append(bs)
            self.data_log[f'ue{ue_idx}_prx_hientai'].append(prx_inst if bs is not None else None)
            self.data_log[f'ue{ue_idx}_speed'].append(self.ue_speeds[ue_idx])
            self.data_log[f'ue{ue_idx}_speed_eff'].append(self.ue_speeds[ue_idx] * (eff_factor if eff_factor>0 else 0))
            self.data_log[f'ue{ue_idx}_white_zone'].append(1 if now_white else 0)
            self.data_log[f'ue{ue_idx}_white_entries'].append(self.white_entries[ue_idx])
            self.data_log[f'ue{ue_idx}_stop_remain'].append(self.stop_counters[ue_idx])

        self.time_text.set_text(f'Bước: {frame}')
        self.fig.canvas.draw(); self.fig.canvas.flush_events()

        if frame == self.steps:
            try: self.save_data_to_csv()
            except Exception as e: print('CSV save failed:', e)

    def save_data_to_csv(self):
        max_len = max((len(v) for v in self.data_log.values()), default=0)
        for k in self.data_log:
            while len(self.data_log[k]) < max_len:
                self.data_log[k].append(None)
        df = pd.DataFrame(self.data_log)
        for col in df.columns:
            if any(tag in col for tag in ['prx', '_x', '_y']):
                df[col] = pd.to_numeric(df[col], errors='coerce').round(1)
        cols = ['Bước']
        for i in range(self.num_ues):
            cols += [f'ue{i}_x', f'ue{i}_y', f'ue{i}_huong', f'ue{i}_BS_ketnoi',
                     f'ue{i}_prx_hientai', f'ue{i}_speed', f'ue{i}_speed_eff',
                     f'ue{i}_white_zone', f'ue{i}_white_entries', f'ue{i}_stop_remain']
        cols = [c for c in cols if c in df.columns]
        df = df.reindex(columns=cols)
        filename = f"du_lieu_cong_suat_nhan_BLACK50.csv"
        df.to_csv(filename, index=False, sep=';', decimal=',', encoding='utf-8-sig', float_format='%.2f')
        print(f"Đã lưu CSV: {filename}")

    def run_animation(self):
        self.animation_running=True; self.current_frame=0
        for frame in range(0, self.steps+1):
            if not self.animation_running: break
            self.update(frame); plt.pause(self.pause_s)
        try: self.save_data_to_csv()
        except Exception as e: print('CSV save failed:', e)

if __name__ == "__main__":
    path_or_url = input("Dán ĐƯỜNG DẪN ẢNH (nền trắng, vùng ĐEN): ").strip()
    try:
        L = float(input("Nhập chiều DÀI HCN (m): ")); W = float(input("Nhập chiều RỘNG HCN (m): "))
    except: L, W = 8000.0, 5000.0
    try: num_ues = int(input("Nhập số UE: "))
    except: num_ues = 5
    try:
        r = input("Tỷ lệ Manhattan (0..1, mặc định 0.7): ").strip()
        ratio = float(r) if r else 0.7
        ratio = min(max(ratio, 0.0), 1.0)
    except: ratio = 0.7
    try:
        t = input("Ngưỡng màu đen (0..255, mặc định 64): ").strip()
        black_th = int(t) if t else 64
    except: black_th = 64

    show_lines = input("Hiện đường UE↔BS? (y/N): ").strip().lower().startswith('y')
    print(f"origin=UPPER | fast_mode=ON | show_link_lines={show_lines} | không legend BS")
    sim = CellularNetworkReceivedPower(
        num_ues=num_ues, manhattan_ratio=ratio,
        image_path_or_url=_clean_path(path_or_url), rect_len_m=L, rect_wid_m=W,
        black_threshold=black_th, black_ratio_los=0.50,
        draw_hex=True, show_link_lines=show_lines, fast_mode=True,
        # Có thể tinh chỉnh các tham số bên dưới nếu cần
        white_win_radius_m=50.0,
        white_density_thresh=0.55,
        white_slow_factor=0.4,
        stop_after_entries=3,
        stop_duration_steps=10
    )
    sim.run_animation(); plt.ioff(); plt.show()
