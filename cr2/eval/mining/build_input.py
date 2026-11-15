import json
import os

CLIPS = ["01d3588e", "26f8dc7e", "69cf2443", "c3ca785e",
         "ecafce84", "76f4a312", "0dd3eb7f", "1f7ef408"]
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "frames")
PROMPT = (
    "You are an offline driving-data mining system. The six images are consecutive frames "
    "from one forward-camera driving clip. Analyze the clip and output ONLY a single JSON "
    "object with exactly these keys and no other text: "
    '{"scene": one of ["highway","urban_street","intersection","parking_lot","other"], '
    '"lighting": one of ["day","night","dusk_or_dawn"], '
    '"weather": one of ["clear","overcast","rain","fog","snow"], '
    '"events": a list of any of ["work_zone","complex_intersection","dense_pedestrians",'
    '"emergency_incident","animals","special_vehicle","cyclists","road_debris"] '
    '(use [] if none), '
    '"summary": one short English sentence on the most safety-relevant aspect of this clip.}'
)

reqs = []
for c in CLIPS:
    content = [{"type": "image", "image": "%s/%s/frame_%d.jpg" % (BASE, c, i)}
               for i in range(1, 7)]
    content.append({"type": "text", "text": PROMPT})
    reqs.append({"messages": [{"role": "user", "content": content}]})

d = {"batch_size": 1, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
     "max_generate_length": 300, "requests": reqs}
out = os.path.join(HERE, "mining_input.json")
with open(out, "w") as f:
    json.dump(d, f, indent=1)
print("built %d requests -> %s" % (len(reqs), out))
