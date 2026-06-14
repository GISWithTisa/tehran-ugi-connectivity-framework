01_preprocess_gee.js
//==========================================================================
// FINAL GI_Z --- Tehran Metropolitan Area
// Robust Ecological Signal (Summer 2015–2025)
// Author: Nahid Nemati
//==========================================================================

// =========================
// CONFIG
// =========================
var MAHALE_ASSET = 'projects/ee-nahidnk1982/assets/Mahale_352';
var START_DATE = '2015-06-01';
var END_DATE = '2025-09-30';
var SCALE = 30;
var TILE_SCALE = 4;

// =========================
// LOAD STUDY AREA
// =========================
var mahale = ee.FeatureCollection(MAHALE_ASSET);
var cityGeom = mahale.geometry();
Map.centerObject(mahale, 11);

// =========================
// LANDSAT NDVI (Summer Composite)
// =========================
function maskLandsatQA(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0) // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0))    // cloud shadow
    .and(qa.bitwiseAnd(1 << 5).eq(0));   // snow
  return img.updateMask(mask);
}

function getLandsat(path) {
  return ee.ImageCollection(path)
    .filterBounds(cityGeom)
    .filterDate(START_DATE, END_DATE)
    .filter(ee.Filter.calendarRange(6,8,'month'))
    .map(maskLandsatQA)
    .select(['SR_B5','SR_B4']);
}

var landsat = getLandsat('LANDSAT/LC08/C02/T1_L2')
  .merge(getLandsat('LANDSAT/LC09/C02/T1_L2'));

var ndvi = landsat.median()
  .normalizedDifference(['SR_B5','SR_B4'])
  .rename('NDVI')
  .clip(cityGeom);

// =========================
// WORLDCOVER (Tree + Water)
// =========================
var wc = ee.Image('ESA/WorldCover/v200/2021')
  .select('Map')
  .clip(cityGeom);

var tree = wc.eq(10);
var water = wc.eq(80);   // includes Chitgar Lake

// =========================
// 150m MOVING WINDOW DENSITY
// =========================
var kernel = ee.Kernel.circle({
  radius: 150,
  units: 'meters',
  normalize: true
});

var treeDensity = tree
  .reduceNeighborhood({
    reducer: ee.Reducer.mean(),
    kernel: kernel
  }).rename('TREE_D');

var waterDensity = water
  .reduceNeighborhood({
    reducer: ee.Reducer.mean(),
    kernel: kernel
  }).rename('WATER_D');

// =========================
// Z-SCORE STANDARDIZATION FUNCTION
// =========================
function zScore(img, bandName) {
  var stats = img.reduceRegion({
    reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true),
    geometry: cityGeom,
    scale: SCALE,
    maxPixels: 1e13,
    tileScale: TILE_SCALE
  });
  
  var mean = ee.Number(stats.get(bandName + '_mean'));
  var std  = ee.Number(stats.get(bandName + '_stdDev'));
  
  return img.subtract(mean).divide(std);
}

// =========================
// STANDARDIZE COMPONENTS
// =========================
var z_ndvi  = zScore(ndvi, 'NDVI');
var z_tree  = zScore(treeDensity, 'TREE_D');
var z_water = zScore(waterDensity, 'WATER_D');

// =========================
// COMBINE COMPONENTS
// =========================
var GI_raw = z_ndvi
  .add(z_tree)
  .add(z_water)
  .divide(3);

// =========================
// FINAL CITY-WIDE STANDARDIZATION
// =========================
var GI_Z = zScore(GI_raw.rename('GI_RAW'), 'GI_RAW')
  .rename('GI_Z')
  .clip(cityGeom);

// =========================
// VISUALIZATION
// =========================
Map.addLayer(GI_Z,
  {min:-2, max:2,
   palette:['8c2d04','d94801','fdd49e','c7e9c0','41ab5d','006d2c']},
  'FINAL GI_Z');

// =========================
// EXPORT
// =========================
Export.image.toDrive({
  image: GI_Z,
  description: 'FINAL_GI_Z_Tehran_2015_2025',
  folder: 'GEE_Exports',
  fileNamePrefix: 'FINAL_GI_Z_Tehran',
  region: cityGeom,
  scale: 30,
  crs: 'EPSG:32639',
  maxPixels: 1e13
});

// ==========================================================================
// FINAL IMP — Impervious Surface Density (Resistance Layer)
// ==========================================================================

// =========================
// CONFIG
// =========================
var MAHALE_ASSET = 'projects/ee-nahidnk1982/assets/Mahale_352';
var SCALE = 30;

// =========================
// LOAD STUDY AREA
// =========================
var mahale = ee.FeatureCollection(MAHALE_ASSET);
var cityGeom = mahale.geometry();
Map.centerObject(mahale, 11);

// =========================
// WORLDCOVER 2021
// =========================
var wc = ee.Image('ESA/WorldCover/v200/2021')
  .select('Map')
  .clip(cityGeom);

// Built-up class = 50
var built = wc.eq(50).rename('BUILT');

// =========================
// 150m MOVING WINDOW DENSITY
// =========================
var kernel = ee.Kernel.circle({
  radius: 150,
  units: 'meters',
  normalize: true
});

var IMP = built.reduceNeighborhood({
  reducer: ee.Reducer.mean(),
  kernel: kernel
}).rename('IMP').clip(cityGeom);

// =========================
// OPTIONAL: Clamp to [0,1] for Safety
// =========================
IMP = IMP.clamp(0,1);

// =========================
// VISUAL CHECK
// =========================
Map.addLayer(IMP,
  {min:0, max:1,
   palette:['f7fbff','6baed6','2171b5','08306b']},
  'FINAL IMP Density');

// =========================
// PRINT STATISTICS
// =========================
var stats = IMP.reduceRegion({
  reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true),
  geometry: cityGeom,
  scale: SCALE,
  maxPixels: 1e13
});

print('IMP Statistics:', stats);

// =========================
// EXPORT
// =========================
Export.image.toDrive({
  image: IMP,
  description: 'FINAL_IMP_Density_Tehran_2021',
  folder: 'GEE_Exports',
  fileNamePrefix: 'FINAL_IMP_Density_Tehran',
  region: cityGeom,
  scale: 30,
  crs: 'EPSG:32639',
  maxPixels: 1e13
});
