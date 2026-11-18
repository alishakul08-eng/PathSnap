import sys
from pathlib import Path
import numpy as np
import folium

sys.path.insert(0, "backend/ml")
import knn_pathsnap as kp
import v2_infer

trip_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "visualisations/predicted_routes")
out_dir.mkdir(parents=True, exist_ok=True)

track, nodes, arcs, route = kp.load_trip(trip_dir)
point_arcs, _ = v2_infer.predict_arrays(track, nodes, arcs)


def seg(a):
    f, t = int(arcs[a, 0]), int(arcs[a, 1])
    return [[float(nodes[f, 1]), float(nodes[f, 0])], [float(nodes[t, 1]), float(nodes[t, 0])]]


lon, lat = track[:, 0], track[:, 1]
m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=14, tiles=None)

pad = 0.003
fx, fy = nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1]
near = np.nonzero((fx > lon.min() - pad) & (fx < lon.max() + pad) &
                  (fy > lat.min() - pad) & (fy < lat.max() + pad))[0]
g = folium.FeatureGroup(name="Road network (grey)")
folium.PolyLine([seg(a) for a in near], color="#9a9a9a", weight=1.5, opacity=0.9).add_to(g)
g.add_to(m)

if route is not None:
    g = folium.FeatureGroup(name="Reference route (green)")
    folium.PolyLine([seg(a) for a in route], color="green", weight=6, opacity=0.7).add_to(g)
    g.add_to(m)

pred = []
for a in point_arcs:
    if a >= 0 and (not pred or pred[-1] != a):
        pred.append(int(a))
g = folium.FeatureGroup(name="Predicted route (red)")
folium.PolyLine([seg(a) for a in pred], color="red", weight=3, opacity=0.9).add_to(g)
g.add_to(m)

step = max(1, len(track) // 1500)
g = folium.FeatureGroup(name="GPS points (blue)")
for x, y in track[::step, :2]:
    folium.CircleMarker([float(y), float(x)], radius=1, weight=1, color="blue", fill=True).add_to(g)
g.add_to(m)

folium.LayerControl().add_to(m)
out = out_dir / f"trip_{trip_dir.name}.html"
m.save(str(out))
print("saved", out)
