// Car layout for a walkaround: panels, the joints between them, and a top-down diagram.
// Left/right as seen from the driver's seat (Uzbekistan drives on the right; the driver sits on the left).

export const PANELS = {
  f_bumper:   { name: "Front bumper",          plastic: true },
  hood:       { name: "Hood" },
  lf_fender:  { name: "Front left fender" },
  rf_fender:  { name: "Front right fender" },
  lf_door:    { name: "Front left door" },
  rf_door:    { name: "Front right door" },
  lr_door:    { name: "Rear left door" },
  rr_door:    { name: "Rear right door" },
  lr_quarter: { name: "Rear left quarter" },
  rr_quarter: { name: "Rear right quarter" },
  trunk:      { name: "Trunk lid" },
  r_bumper:   { name: "Rear bumper",           plastic: true },
  roof:       { name: "Roof" },
};

// Walkaround order: start at the front left corner and go clockwise seen from above (down the left side, back up the right).
// `a` is the panel on the left of the gap in the photo, `b` on the right, when you face the car.
export const JOINTS = [
  { id: "j1",  a: "f_bumper",   b: "lf_fender",  where: "Front left corner, bumper to fender" },
  { id: "j2",  a: "lf_fender",  b: "hood",       where: "Hood edge, left side" },
  { id: "j3",  a: "lf_fender",  b: "lf_door",    where: "Left side, fender to front door" },
  { id: "j4",  a: "lf_door",    b: "lr_door",    where: "Left side, front door to rear door" },
  { id: "j5",  a: "lr_door",    b: "lr_quarter", where: "Left side, rear door to quarter" },
  { id: "j6",  a: "lr_quarter", b: "r_bumper",   where: "Rear left corner, quarter to bumper" },
  { id: "j7",  a: "lr_quarter", b: "trunk",      where: "Trunk edge, left side" },
  { id: "j8",  a: "trunk",      b: "rr_quarter", where: "Trunk edge, right side" },
  { id: "j9",  a: "r_bumper",   b: "rr_quarter", where: "Rear right corner, bumper to quarter" },
  { id: "j10", a: "rr_quarter", b: "rr_door",    where: "Right side, quarter to rear door" },
  { id: "j11", a: "rr_door",    b: "rf_door",    where: "Right side, rear door to front door" },
  { id: "j12", a: "rf_door",    b: "rf_fender",  where: "Right side, front door to fender" },
  { id: "j13", a: "hood",       b: "rf_fender",  where: "Hood edge, right side" },
  { id: "j14", a: "rf_fender",  b: "f_bumper",   where: "Front right corner, fender to bumper" },
];

// Some joints differ more with the same paint, so they need a bigger difference before we call them.
// Fitted on real photos so that original paint is flagged about as often at every kind of joint (eval/README):
//  - bumpers are plastic, painted separately at the factory, and sit at an angle to the body: 1.4
//  - hood and trunk edges: the hood faces the sky, the fender faces sideways, so they reflect different light: 1.8
export const BUMPER_FACTOR = 1.4, LID_FACTOR = 1.8;
const LIDS = new Set(["hood", "trunk"]);
export const jointFactor = (j) =>
  LIDS.has(j.a) || LIDS.has(j.b) ? LID_FACTOR : PANELS[j.a].plastic || PANELS[j.b].plastic ? BUMPER_FACTOR : 1;

// Top-down diagram, front at the top. viewBox 0 0 220 440.
export const SHAPES = {
  f_bumper:   "M52 14 Q110 2 168 14 L176 40 L44 40 Z",
  hood:       "M50 42 L170 42 L164 126 L56 126 Z",
  lf_fender:  "M30 44 L48 42 L54 126 L28 126 Q24 84 30 44 Z",
  rf_fender:  "M190 44 L172 42 L166 126 L192 126 Q196 84 190 44 Z",
  lf_door:    "M28 130 L52 130 L50 212 L26 212 Z",
  lr_door:    "M26 216 L50 216 L50 290 L27 290 Z",
  lr_quarter: "M27 294 L52 294 L56 392 L34 392 Q26 340 27 294 Z",
  rf_door:    "M192 130 L168 130 L170 212 L194 212 Z",
  rr_door:    "M194 216 L170 216 L170 290 L193 290 Z",
  rr_quarter: "M193 294 L168 294 L164 392 L186 392 Q194 340 193 294 Z",
  roof:       "M60 176 L160 176 L160 286 L60 286 Z",
  trunk:      "M58 330 L162 330 L160 392 L60 392 Z",
  r_bumper:   "M42 396 L178 396 L170 424 Q110 434 50 424 Z",
};
export const GLASS = ["M58 130 L162 130 L158 172 L62 172 Z", "M62 290 L158 290 L162 326 L58 326 Z"];
export const JOINT_XY = {
  j1: [38, 40], j2: [52, 84], j3: [40, 128], j4: [38, 214], j5: [38, 292], j6: [44, 394], j7: [56, 360],
  j8: [164, 360], j9: [176, 394], j10: [182, 292], j11: [182, 214], j12: [180, 128], j13: [168, 84], j14: [182, 40],
};
