import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

// Run with the presentation runtime. Dependencies and output paths are explicit
// so that this source contains no machine-specific settings.
const [sourceDir, buildDir, packageDir] = process.argv.slice(2);
if (!sourceDir || !buildDir || !packageDir) {
  throw new Error("Usage: node build.mjs <source-directory> <build-directory> <node-packages>");
}
const { Presentation, PresentationFile } = await import(pathToFileURL(
  path.join(packageDir, "@oai", "artifact-tool", "dist", "artifact_tool.mjs")
).href);
const content = JSON.parse(await fs.readFile(path.join(sourceDir, "content.json"), "utf8"));
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
await fs.mkdir(buildDir, { recursive: true });
const c = {
  dark: "#103E49", ink: "#154752", paper: "#F6F5F0", cream: "#F7F5E9",
  body: "#CDE1DF", gold: "#E5C989", muted: "#526C70", mint: "#92CCC5"
};
function text(slide, name, value, x, y, w, h, size, color, bold = false) {
  const shape = slide.shapes.add({
    geometry: "textbox", name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { fill: "none", width: 0, style: "solid" }
  });
  shape.text = value;
  shape.text.style = {
    typeface: "Calibri", fontSize: size, color, bold,
    insets: 0, verticalAlignment: "top", wrap: "none", autoFit: "none"
  };
  return shape;
}
function area(slide, name, x, y, w, h, fill) {
  return slide.shapes.add({
    geometry: "rect", name,
    position: { left: x, top: y, width: w, height: h },
    fill, line: { fill: "none", width: 0, style: "solid" }
  });
}
const s1 = presentation.slides.add();
s1.background.fill = c.dark;
const a = content.slides[0];
text(s1, "title", a.title, 64, 44, 1152, 62, 48, c.cream, true);
text(s1, "subtitle", a.subtitle.replace(" for people", "\nfor people"), 64, 119, 1152, 70, 26, c.body);
const bytes = await fs.readFile(path.join(sourceDir, "assets", "feature_creation.png"));
s1.images.add({
  blob: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  contentType: "image/png",
  alt: "Conceptual illustration of historical records becoming measurable features and entering a neural model.",
  fit: "cover",
  position: { left: 0, top: 187, width: 1280, height: 267 }
});
const cols = [64, 480, 896];
for (let i = 0; i < 3; i++) {
  text(s1, `stage-${i + 1}`, a.stages[i].heading, cols[i], 463, 342, 42, 29, c.gold, true);
  text(s1, `description-${i + 1}`, a.stages[i].body, cols[i], 508, 350, 60, 23, c.cream);
}
text(s1, "comparison", a.comparison, 64, 586, 1152, 35, 24, c.body);
area(s1, "status-band", 0, 635, 1280, 85, "#0B303A");
text(s1, "status-heading", a.statusHeading, 64, 652, 205, 38, 26, c.gold, true);
text(s1, "status", a.status, 294, 646, 900, 62, 24, c.cream);
s1.speakerNotes.text = a.notes;

const s2 = presentation.slides.add();
s2.background.fill = c.paper;
const b = content.slides[1];
text(s2, "title", b.title, 64, 44, 1152, 62, 48, c.ink, true);
text(s2, "subtitle", b.subtitle, 64, 119, 1152, 42, 26, c.muted);
for (let i = 0; i < 4; i++) {
  const x = 64 + 292 * i;
  const stage = b.stages[i];
  text(s2, `number-${i + 1}`, stage.number, x, 210, 260, 65, 56, "#81B5B0", true);
  text(s2, `stage-${i + 1}`, stage.heading, x, 283, 276, 43, 29, c.ink, true);
  text(s2, `description-${i + 1}`, stage.body, x, 334, 275, 97, 24, c.muted);
}
text(s2, "reuse-heading", b.reuseHeading, 64, 467, 532, 40, 28, c.ink, true);
text(s2, "reuse", b.reuseBody, 64, 512, 550, 65, 24, c.muted);
text(s2, "configure-heading", b.configureHeading, 648, 467, 568, 40, 28, c.ink, true);
text(s2, "configure", b.configureBody, 648, 512, 568, 65, 24, c.muted);
area(s2, "milestone-band", 0, 606, 1280, 114, c.dark);
text(s2, "milestone-heading", b.roadmapHeading, 64, 625, 220, 40, 26, c.gold, true);
text(s2, "milestone", b.roadmap, 294, 627, 922, 38, 24, c.cream);
text(s2, "intended-value", b.value, 64, 672, 1152, 28, 21, c.body);
s2.speakerNotes.text = b.notes;

for (let i = 0; i < presentation.slides.items.length; i++) {
  const slide = presentation.slides.items[i];
  const preview = await presentation.export({ slide, format: "png", scale: 1.5 });
  await fs.writeFile(path.join(buildDir, `slide-${i + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(buildDir, `slide-${i + 1}.layout.json`), await layout.text());
}
await (await PresentationFile.exportPptx(presentation)).save(path.join(buildDir, "candidate.pptx"));
console.log(JSON.stringify({ slides: presentation.slides.items.length, candidate: path.join(buildDir, "candidate.pptx") }));
