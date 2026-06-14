// ==========================================================================
// PEARSON CORRELATION BETWEEN GI_Z AND IMP
// ==========================================================================

var GI = ee.Image('projects/ee-nahidnk1982/assets/GI_Z').rename('GI');
var IMP = ee.Image('projects/ee-nahidnk1982/assets/IMP').rename('IMP');

var stack = GI.addBands(IMP);

var corr = stack.reduceRegion({
  reducer: ee.Reducer.pearsonsCorrelation(),
  geometry: ee.FeatureCollection('projects/ee-nahidnk1982/assets/Mahale_352').geometry(),
  scale: 30,
  maxPixels: 1e13,
  tileScale: 4
});

print('Pearson Correlation GI vs IMP:', corr);