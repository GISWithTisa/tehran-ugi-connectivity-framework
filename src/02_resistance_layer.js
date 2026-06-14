// ==========================================================================
// FINAL ECOLOGICAL RESISTANCE — Percentile Normalization (Robust)
// ==========================================================================

var SCALE = 30;
var TILE_SCALE = 4;

var city = ee.FeatureCollection('projects/ee-nahidnk1982/assets/Mahale_352')
              .geometry();

var GI  = ee.Image('projects/ee-nahidnk1982/assets/GI_Z')
              .rename('GI')
              .clip(city);

var IMP = ee.Image('projects/ee-nahidnk1982/assets/IMP')
              .rename('IMP')
              .clip(city);

// ===============================
// ROBUST NORMALIZATION (2–98 Percentile)
// ===============================

var giPercentiles = GI.reduceRegion({
  reducer: ee.Reducer.percentile([2,98]),
  geometry: city,
  scale: SCALE,
  maxPixels: 1e13,
  tileScale: TILE_SCALE
});

var p2  = ee.Number(giPercentiles.get('GI_p2'));
var p98 = ee.Number(giPercentiles.get('GI_p98'));

print('GI 2nd percentile:', p2);
print('GI 98th percentile:', p98);

// Normalize using percentiles
var GI_scaled = GI.unitScale(p2, p98).clamp(0,1);

var GI_inv = ee.Image(1).subtract(GI_scaled);

// ===============================
// NONLINEAR TRANSFORMATION
// ===============================

var IMP_nl = IMP.pow(2);
var GI_nl  = GI_inv.pow(2);

// ===============================
// COMBINE
// ===============================

var resistance = IMP_nl.multiply(0.5)
  .add(GI_nl.multiply(0.5))
  .rename('RESISTANCE')
  .clamp(0,1)
  .clip(city);

// ===============================
// VISUALIZATION
// ===============================

Map.centerObject(city, 11);

Map.addLayer(resistance,
  {min:0, max:1,
   palette:['00ff00','ffff00','ff8800','ff0000']},
  'FINAL Ecological Resistance');

// ===============================
// STATISTICS
// ===============================

var resStats = resistance.reduceRegion({
  reducer: ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), '', true)
            .combine(ee.Reducer.minMax(), '', true),
  geometry: city,
  scale: SCALE,
  maxPixels: 1e13,
  tileScale: TILE_SCALE
});

print('FINAL Resistance Statistics:', resStats);

// ===============================
// EXPORT
// ===============================

Export.image.toDrive({
  image: resistance,
  description: 'FINAL_Ecological_Resistance_Tehran_Robust',
  folder: 'GEE_Exports',
  fileNamePrefix: 'FINAL_Ecological_Resistance_Tehran_Robust',
  region: city,
  scale: 30,
  crs: 'EPSG:32639',
  maxPixels: 1e13
});