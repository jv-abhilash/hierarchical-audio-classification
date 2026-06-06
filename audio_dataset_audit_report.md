# Audio Dataset Audit

Note: Step 4 sample checks use librosa.load(sr=None, mono=False). Bulk duration/slice estimates use WAV headers for speed.


## rain_dataset

### Step 1 - Identify Dataset

Folder structure up to 3 levels deep:

- None

Root non-audio files: rain_dataset/weather.csv

First 10 lines of `rain_dataset/weather.csv`:
```text
"Data.Precipitation","Date.Full","Date.Month","Date.Week of","Date.Year","Station.City","Station.Code","Station.Location","Station.State","Data.Temperature.Avg Temp","Data.Temperature.Max Temp","Data.Temperature.Min Temp","Data.Wind.Direction","Data.Wind.Speed"
"0.0","2016-01-03","1","3","2016","Birmingham","BHM","Birmingham, AL","Alabama","39","46","32","33","4.33"
"0.0","2016-01-03","1","3","2016","Huntsville","HSV","Huntsville, AL","Alabama","39","47","31","32","3.86"
"0.16","2016-01-03","1","3","2016","Mobile","MOB","Mobile, AL","Alabama","46","51","41","35","9.73"
"0.0","2016-01-03","1","3","2016","Montgomery","MGM","Montgomery, AL","Alabama","45","52","38","32","6.86"
"0.01","2016-01-03","1","3","2016","Anchorage","ANC","Anchorage, AK","Alaska","34","38","29","19","7.8"
"0.09","2016-01-03","1","3","2016","Annette","ANN","Annette, AK","Alaska","38","44","31","9","8.7"
"0.05","2016-01-03","1","3","2016","Bethel","BET","Bethel, AK","Alaska","30","36","24","9","16.46"
"0.15","2016-01-03","1","3","2016","Bettles","BTT","Bettles, AK","Alaska","22","32","9","2","3.1"
"0.6","2016-01-03","1","3","2016","Cold Bay","CDB","Cold Bay, AK","Alaska","34","36","31","20","9.1"
```
Subfolder names: None

### Step 2 - Count Audio Files

Total requested-extension audio files: 0 (`wav/mp3/flac/ogg/m4a/aiff`). Including WebM/nonstandard: 0.
Sample filenames:
- None
Unique parent folder names containing requested audio: None

### Step 3 - Inspect Metadata Files

Metadata `rain_dataset/weather.csv`
- Columns/keys: ['Data.Precipitation', 'Date.Full', 'Date.Month', 'Date.Week of', 'Date.Year', 'Station.City', 'Station.Code', 'Station.Location', 'Station.State', 'Data.Temperature.Avg Temp', 'Data.Temperature.Max Temp', 'Data.Temperature.Min Temp', 'Data.Wind.Direction', 'Data.Wind.Speed']
- Total row/key count: 16743
- First 10 rows/lines:
```text
{'Data.Precipitation': '0.0', 'Date.Full': '2016-01-03', 'Date.Month': '1', 'Date.Week of': '3', 'Date.Year': '2016', 'Station.City': 'Birmingham', 'Station.Code': 'BHM', 'Station.Location': 'Birmingham, AL', 'Station.State': 'Alabama', 'Data.Temperature.Avg Temp': '39', 'Data.Temperature.Max Temp': '46', 'Data.Temperature.Min Temp': '32', 'Data.Wind.Direction': '33', 'Data.Wind.Speed': '4.33'}
{'Data.Precipitation': '0.0', 'Date.Full': '2016-01-03', 'Date.Month': '1', 'Date.Week of': '3', 'Date.Year': '2016', 'Station.City': 'Huntsville', 'Station.Code': 'HSV', 'Station.Location': 'Huntsville, AL', 'Station.State': 'Alabama', 'Data.Temperature.Avg Temp': '39', 'Data.Temperature.Max Temp': '47', 'Data.Temperature.Min Temp': '31', 'Data.Wind.Direction': '32', 'Data.Wind.Speed': '3.86'}
{'Data.Precipitation': '0.16', 'Date.Full': '2016-01-03', 'Date.Month': '1', 'Date.Week of': '3', 'Date.Year': '2016', 'Station.City': 'Mobile', 'Station.Code': 'MOB', 'Station.Location': 'Mobile, AL', 'Station.State': 'Alabama', 'Data.Temperature.Avg Temp': '46', 'Data.Temperature.Max Temp': '51', 'Data.Temperature.Min Temp': '41', 'Data.Wind.Direction': '35', 'Data.Wind.Speed': '9.73'}
{'Data.Precipitation': '0.0', 'Date.Full': '2016-01-03', 'Date.Month': '1', 'Date.Week of': '3', 'Date.Year': '2016', 'Station.City': 'Montgomery', 'Station.Code': 'MGM', 'Station.Location': 'Montgomery, AL', 'Station.State': 'Alabama', 'Data.Temperature.Avg Temp': '45', 'Data.Temperature.Max Temp': '52', 'Data.Temperature.Min Temp': '38', 'Data.Wind.Direction': '32', 'Data.Wind.Speed': '6.86'}
{'Data.Precipitation': '0.01', 'Date.Full': '2016-01-03', 'Date.Month': '1', 'Date.Week of': '3', 'Date.Year': '2016', 'Station.City': 'Anchorage', 'Station.Code': 'ANC', 'Station.Location': 'Anchorage, AK', 'Station.State': 'Alaska', 'Data.Temperature.Avg Temp': '34', 'Data.Temperature.Max Temp': '38', 'Data.Temperature.Min Temp': '29', 'Data.Wind.Direction': '19', 'Data.Wind.Speed': '7.8'}
{'Data.Precipitation': '0.09', 'Date.Full': '2016-01-03', 'Date.Month': '1', 'Date.Week of': '3', 'Date.Year': '2016', 'Station.City': 'Annette', 'Station.Code': 'ANN', 'Station.Location': 'Annette, AK', 'Station.State': 'Alaska', 'Data.Temperature.Avg Temp': '38', 'Data.Temperature.Max Temp': '44', 'Data.Temperature.Min Temp': '31', 'Data.Wind.Direction': '9', 'Data.Wind.Speed': '8.7'}
{'Data.Precipitation': '0.05', 'Date.Full': '2016-01-03', 'Date.Month': '1', 'Date.Week of': '3', 'Date.Year': '2016', 'Station.City': 'Bethel', 'Station.Code': 'BET', 'Station.Location': 'Bethel, AK', 'Station.State': 'Alaska', 'Data.Temperature.Avg Temp': '30', 'Data.Temperature.Max Temp': '36', 'Data.Temperature.Min Temp': '24', 'Data.Wind.Direction': '9', 'Data.Wind.Speed': '16.46'}
{'Data.Precipitation': '0.15', 'Date.Full': '2016-01-03', 'Date.Month': '1', 'Date.Week of': '3', 'Date.Year': '2016', 'Station.City': 'Bettles', 'Station.Code': 'BTT', 'Station.Location': 'Bettles, AK', 'Station.State': 'Alaska', 'Data.Temperature.Avg Temp': '22', 'Data.Temperature.Max Temp': '32', 'Data.Temperature.Min Temp': '9', 'Data.Wind.Direction': '2', 'Data.Wind.Speed': '3.1'}
{'Data.Precipitation': '0.6', 'Date.Full': '2016-01-03', 'Date.Month': '1', 'Date.Week of': '3', 'Date.Year': '2016', 'Station.City': 'Cold Bay', 'Station.Code': 'CDB', 'Station.Location': 'Cold Bay, AK', 'Station.State': 'Alaska', 'Data.Temperature.Avg Temp': '34', 'Data.Temperature.Max Temp': '36', 'Data.Temperature.Min Temp': '31', 'Data.Wind.Direction': '20', 'Data.Wind.Speed': '9.1'}
{'Data.Precipitation': '2.15', 'Date.Full': '2016-01-03', 'Date.Month': '1', 'Date.Week of': '3', 'Date.Year': '2016', 'Station.City': 'Cordova', 'Station.Code': 'CDV', 'Station.Location': 'Cordova, AK', 'Station.State': 'Alaska', 'Data.Temperature.Avg Temp': '38', 'Data.Temperature.Max Temp': '43', 'Data.Temperature.Min Temp': '33', 'Data.Wind.Direction': '9', 'Data.Wind.Speed': '9.76'}
```
### Step 4 - Audio Quality Check

No requested-extension audio files to sample.

Sample-rate counts from sampled files: {}; formats: {}

### Step 5 - Nature Subclass Mapping

- rain: NOT PRESENT
- sea_waves: NOT PRESENT
- wind: NOT PRESENT
- crackling_fire: NOT PRESENT

### Step 7 - Problem Flags

- total useful 5s clips after filtering below 100
Decision: SKIP


## audio_noise_dataset

### Step 1 - Identify Dataset

Folder structure up to 3 levels deep:

- None

Root non-audio files: audio_noise_dataset/sample-1.webm, audio_noise_dataset/sample-10.webm, audio_noise_dataset/sample-2.webm, audio_noise_dataset/sample-3.webm, audio_noise_dataset/sample-4.webm, audio_noise_dataset/sample-5.webm, audio_noise_dataset/sample-6.webm, audio_noise_dataset/sample-7.webm, audio_noise_dataset/sample-8.webm, audio_noise_dataset/sample-9.webm

Subfolder names: None

### Step 2 - Count Audio Files

Total requested-extension audio files: 0 (`wav/mp3/flac/ogg/m4a/aiff`). Including WebM/nonstandard: 10.
Sample filenames:
- None
Unique parent folder names containing requested audio: None

### Step 3 - Inspect Metadata Files

No CSV/JSON/txt/md metadata files found.

### Step 4 - Audio Quality Check

No requested-extension audio files to sample.

Sample-rate counts from sampled files: {}; formats: {'webm': 10}

### Step 5 - Nature Subclass Mapping

- rain: NOT PRESENT
- sea_waves: NOT PRESENT
- wind: NOT PRESENT
- crackling_fire: NOT PRESENT

### Step 7 - Problem Flags

- total useful 5s clips after filtering below 100
- non-standard or less common formats present: {'webm': 10}
- no metadata or label mechanism found
- only non-requested audio extension found
Decision: SKIP


## FSC22_forest

### Step 1 - Identify Dataset

Folder structure up to 3 levels deep:

- `Audio Wise V1.0-20220916T202003Z-001/`
- `  Audio Wise V1.0/`
- `Metadata-20220916T202011Z-001/`
- `  Metadata/`

Root non-audio files: None

Subfolder names: Audio Wise V1.0-20220916T202003Z-001, Metadata-20220916T202011Z-001

### Step 2 - Count Audio Files

Total requested-extension audio files: 2025 (`wav/mp3/flac/ogg/m4a/aiff`). Including WebM/nonstandard: 2025.
Sample filenames:
- `FSC22_forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0/21_12172.wav` parent `FSC22_forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0`
- `FSC22_forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0/17_11726.wav` parent `FSC22_forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0`
- `FSC22_forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0/26_12672.wav` parent `FSC22_forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0`
- `FSC22_forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0/23_12334.wav` parent `FSC22_forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0`
- `FSC22_forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0/27_12715.wav` parent `FSC22_forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0`
Unique parent folder names containing requested audio: Audio Wise V1.0

### Step 3 - Inspect Metadata Files

Metadata `FSC22_forest/Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv`
- Columns/keys: ['Source File Name', 'Dataset File Name', 'Class ID', 'Class Name']
- Total row/key count: 2025
- Unique label-like values in `Source File Name`: 04-1 Chainsaw.wav.wav, 080320 Chainsaw start, branches falling and engine cut.wav.wav, 100118__noisecollector__raw-squirrelbark.wav, 100119__noisecollector__raw-squirrelbarker.wav, 100277__soundbytez__squirrel-monkeys01.wav, 100278__soundbytez__squirrel-monkeys02.wav, 100784__tomlija__pedestrian-zone-knez-mihajlova-street.wav, 102549__tomlija__train-station-lobby-ambience-with-a-pa-announcer.wav, 102953_1.wav, 102953_2.wav, 102954_1.wav, 102954_2.wav, 103140__hoersturz__20100624-102252-soda-butte-squirrel.wav, 103169__fons__synth-clap.wav, 103382__miastodzwiekow__in-front-of-block220810.wav, 103598__floating-tree__walking-in-the-lone-shieling.wav, 103599__floating-tree__walking-lone-shieling-quack.wav, 103601__floating-tree__wet-stone-walking.wav, 104932__dobroide__20100918-starling-fluttering-02.wav, 104933__dobroide__20100918-starling-fluttering-03-2.wav, 104933__dobroide__20100918-starling-fluttering-03.wav, 104934__dobroide__20100918-starling-fluttering.wav, 104952_1.wav, 104952_2.wav, 104952_3.wav, 104952_4.wav, 104956_A.wav, 104956_B.wav, 104956_C.wav, 104956_D.wav, 104956_E.wav, 105954_3.wav, 106016__chgrasse__gf.wav, 106092__noisecollector__squirelwar1.wav, 106134_1.wav, 106134_2.wav, 106134_3.wav, 106554__tomlija__street-rally-walla-chatter-noises-whistles.wav, 106819__plusminuszero__talking-and-laughter.wav, 108162__burkay__whistle-human.wav, 109736__soundcollectah__double-hit.wav, 109738__soundcollectah__hilow-multi-hits.wav, 109741__soundcollectah__hit-loud.wav, 109752__soundcollectah__tre-hit-01.wav, 109753__soundcollectah__tre-hit-long.wav, 110613__soundscalpel-com__animals-insect-cricket-001.wav, 110919_1.wav, 110919_2.wav, 110919_3.wav, 110919_4.wav, 111094__soundbytez__roseate-spoonbill02-2.wav, 111094__soundbytez__roseate-spoonbill02.wav, 111121__donjupitor__ste-000-2.wav, 111121__donjupitor__ste-000.wav, 111393__philberts__red-squirrel-chatter.wav, 114594__herbertboland__generator1.wav, 114595__herbertboland__generator2.wav, 114596__herbertboland__generator3.wav, 115545__matucha__fireworks-2009-close.wav, 11597_1.wav, 11597_2.wav, 116172__jacobsteel__short-whistles.wav, 117114__heigh-hoo__generate.wav, 117350__stereostereo__13-sparrow-real.wav, 117351__stereostereo__14-sparrow-foley.wav, 117548__jppi-stu__sw-frogsandplane.wav, 117615__soundmary__fireworks-explode-and-fizz.wav, 117616__soundmary__fireworks-exploding-1.wav, 117617__soundmary__fireworks.wav, 117627__soundmary__footsteps-gravel-pavement.wav, 117628__soundmary__footsteps-on-stone-patio-1.wav, 117629__soundmary__footsteps-on-stone-patio.wav, 117630__soundmary__footsteps-woodland.wav, 118450__omar-alvarado__gunshot.wav, 118453__omar-alvarado__helicopter-leave.wav, 119648__suso-ramallo__fireworks-3-16.wav, 119649__suso-ramallo__fireworks-4-16.wav, 11Chainsaw.wav.wav, 120210_1__sportygurl37__clapping-2.wav, 120210_2__sportygurl37__clapping-2-2.wav, 120210__sportygurl37__clapping.wav, 120358__cemagar__whistling.wav, 120695.wav, 120698.wav, 120699.wav, 120701.wav, 120808.wav, 122260__echobones__angry-squirrel-long.wav, 123805__kendallbear__clap-for-fred.wav, 124015__alienistcog__electronic-buzz1.wav, 125940__juskiddink__beating-wings-2.wav, 125940__juskiddink__beating-wings.wav, 126448__inchadney__helicopter.wav, 126910_1.wav, 126910_2.wav, 126913_1.wav, 126913_2.wav, 128134__be-steele__boris-squeal-moan.wav, 128469_(1)_be-steele__igor-three-howls.wav, 128469_(2)_be-steele__igor-three-howls.wav, 128469__be-steele__igor-three-howls.wav, 130326_1__donut-kiss__clapping-and-yelling-2.wav, 130326__donut-kiss__clapping-and-yelling.wav, 130458__mmiron__lake-highway-birds-frogs-1.wav, 130459__mmiron__lake-highway-birds-frogs.wav, 132151_1.wav, 132151_2.wav, 132151_3.wav, 133500__setuniman__circling-helicopter-m97-34c.wav, 133675__klangpinnwand__helicopter-start.wav, 133763_(1)_klankbeeld__werewolf-01.wav, 133763_(2)_klankbeeld__werewolf-01.wav, 133763_(3)_klankbeeld__werewolf-01.wav, 133763_(4)_klankbeeld__werewolf-01.wav, 133763_(5)_klankbeeld__werewolf-01.wav, 133763_(6)_klankbeeld__werewolf-01.wav, 134909.wav, 134944__kvgarlic__summerinsectscardinal.wav, 134950__kvgarlic__summerinsectchorus.wav, 13533__acclivity__bumblebee.wav ... (+180 more)
- Unique label-like values in `Dataset File Name`: 10_11001.wav, 10_11002.wav, 10_11003.wav, 10_11004.wav, 10_11005.wav, 10_11006.wav, 10_11007.wav, 10_11008.wav, 10_11009.wav, 10_11010.wav, 10_11011.wav, 10_11012.wav, 10_11013.wav, 10_11014.wav, 10_11015.wav, 10_11016.wav, 10_11017.wav, 10_11018.wav, 10_11019.wav, 10_11020.wav, 10_11021.wav, 10_11022.wav, 10_11023.wav, 10_11024.wav, 10_11025.wav, 10_11026.wav, 10_11027.wav, 10_11028.wav, 10_11029.wav, 10_11030.wav, 10_11031.wav, 10_11032.wav, 10_11033.wav, 10_11034.wav, 10_11035.wav, 10_11036.wav, 10_11037.wav, 10_11038.wav, 10_11039.wav, 10_11040.wav, 10_11041.wav, 10_11042.wav, 10_11043.wav, 10_11044.wav, 10_11045.wav, 10_11046.wav, 10_11047.wav, 10_11048.wav, 10_11049.wav, 10_11050.wav, 10_11051.wav, 10_11052.wav, 10_11053.wav, 10_11054.wav, 10_11055.wav, 10_11056.wav, 10_11057.wav, 10_11058.wav, 10_11059.wav, 10_11060.wav, 10_11061.wav, 10_11062.wav, 10_11063.wav, 10_11064.wav, 10_11065.wav, 10_11066.wav, 10_11067.wav, 10_11068.wav, 10_11069.wav, 10_11070.wav, 10_11071.wav, 10_11072.wav, 10_11073.wav, 10_11074.wav, 10_11075.wav, 11_11101.wav, 11_11102.wav, 11_11103.wav, 11_11104.wav, 11_11105.wav, 11_11106.wav, 11_11107.wav, 11_11108.wav, 11_11109.wav, 11_11110.wav, 11_11111.wav, 11_11112.wav, 11_11113.wav, 11_11114.wav, 11_11115.wav, 11_11116.wav, 11_11117.wav, 11_11118.wav, 11_11119.wav, 11_11120.wav, 11_11121.wav, 11_11122.wav, 11_11123.wav, 11_11124.wav, 11_11125.wav, 11_11126.wav, 11_11127.wav, 11_11128.wav, 11_11129.wav, 11_11130.wav, 11_11131.wav, 11_11132.wav, 11_11133.wav, 11_11134.wav, 11_11135.wav, 11_11136.wav, 11_11137.wav, 11_11138.wav, 11_11139.wav, 11_11140.wav, 11_11141.wav, 11_11142.wav, 11_11143.wav, 11_11144.wav, 11_11145.wav ... (+180 more)
- Unique label-like values in `Class ID`: 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 3, 4, 5, 6, 7, 8, 9
- Unique label-like values in `Class Name`: Axe, BirdChirping, Chainsaw, Clapping, Fire, Firework, Footsteps, Frog, Generator, Gunshot, Handsaw, Helicopter, Insect, Lion, Rain, Silence, Speaking, Squirrel, Thunderstorm, TreeFalling, VehicleEngine, WaterDrops, Whistling, Wind, WingFlaping, WolfHowl, WoodChop
- First 10 rows/lines:
```text
{'Source File Name': '17548__A.wav', 'Dataset File Name': '1_10101.wav', 'Class ID': '1', 'Class Name': 'Fire'}
{'Source File Name': '17548_B.wav', 'Dataset File Name': '1_10102.wav', 'Class ID': '1', 'Class Name': 'Fire'}
{'Source File Name': '17548_C.wav', 'Dataset File Name': '1_10103.wav', 'Class ID': '1', 'Class Name': 'Fire'}
{'Source File Name': '17548_D.wav', 'Dataset File Name': '1_10104.wav', 'Class ID': '1', 'Class Name': 'Fire'}
{'Source File Name': '17548_E.wav', 'Dataset File Name': '1_10105.wav', 'Class ID': '1', 'Class Name': 'Fire'}
{'Source File Name': '17548_F.wav', 'Dataset File Name': '1_10106.wav', 'Class ID': '1', 'Class Name': 'Fire'}
{'Source File Name': '17548_G.wav', 'Dataset File Name': '1_10107.wav', 'Class ID': '1', 'Class Name': 'Fire'}
{'Source File Name': '17548_H.wav', 'Dataset File Name': '1_10108.wav', 'Class ID': '1', 'Class Name': 'Fire'}
{'Source File Name': '17717_A.wav', 'Dataset File Name': '1_10109.wav', 'Class ID': '1', 'Class Name': 'Fire'}
{'Source File Name': '17717_B.wav', 'Dataset File Name': '1_10110.wav', 'Class ID': '1', 'Class Name': 'Fire'}
```
### Step 4 - Audio Quality Check

- `FSC22_forest/Audio Wise V1.0-20220916T202003Z-001/Audio Wise V1.0/10_11001.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
Sample-rate counts from sampled files: {'44100': 121, '48000': 85, '24000': 29, '16000': 2, '96000': 13}; formats: {'wav': 2025}

### Step 5 - Nature Subclass Mapping

- rain: FSC22_forest/Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv Class Name in ['Rain']; clips=75; estimated 5s clips=75; sample_rates={'44100': 75}
- sea_waves: NOT PRESENT
- wind: FSC22_forest/Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv Class Name in ['Wind']; clips=75; estimated 5s clips=75; sample_rates={'48000': 75}
- crackling_fire: FSC22_forest/Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv Class Name in ['Fire']; clips=75; estimated 5s clips=75; sample_rates={'44100': 45, '48000': 30}

### Step 7 - Problem Flags

- clipping observed in sampled audio
Decision: USE WITH CAUTION


## forest_wild_fire_sound_dataset

### Step 1 - Identify Dataset

Folder structure up to 3 levels deep:

- `drive-download-20210504T145639Z-001/`
- `  Split Files/`
- `  split file 3/`
- `  split files 2/`
- `split audio 4-20210504T123341Z-001/`
- `  split audio 4/`

Root non-audio files: None

Subfolder names: drive-download-20210504T145639Z-001, split audio 4-20210504T123341Z-001

### Step 2 - Count Audio Files

Total requested-extension audio files: 289 (`wav/mp3/flac/ogg/m4a/aiff`). Including WebM/nonstandard: 289.
Sample filenames:
- `forest_wild_fire_sound_dataset/split audio 4-20210504T123341Z-001/split audio 4/videoplayback (1)_26.wav` parent `forest_wild_fire_sound_dataset/split audio 4-20210504T123341Z-001/split audio 4`
- `forest_wild_fire_sound_dataset/drive-download-20210504T145639Z-001/split file 3/videoplayback_65.wav` parent `forest_wild_fire_sound_dataset/drive-download-20210504T145639Z-001/split file 3`
- `forest_wild_fire_sound_dataset/split audio 4-20210504T123341Z-001/split audio 4/videoplayback (1)_54.wav` parent `forest_wild_fire_sound_dataset/split audio 4-20210504T123341Z-001/split audio 4`
- `forest_wild_fire_sound_dataset/drive-download-20210504T145639Z-001/split files 2/y2mate.com - DD Ambience  Building on Fire  Blaze Inferno Wood Cracking Collapsing Loud Stressing_1_35.wav` parent `forest_wild_fire_sound_dataset/drive-download-20210504T145639Z-001/split files 2`
- `forest_wild_fire_sound_dataset/drive-download-20210504T145639Z-001/split file 3/videoplayback_01.wav` parent `forest_wild_fire_sound_dataset/drive-download-20210504T145639Z-001/split file 3`
Unique parent folder names containing requested audio: Split Files, split audio 4, split file 3, split files 2

### Step 3 - Inspect Metadata Files

No CSV/JSON/txt/md metadata files found.

### Step 4 - Audio Quality Check

- `forest_wild_fire_sound_dataset/drive-download-20210504T145639Z-001/Split Files/y2mate.com - DD Ambience  Building on Fire  Blaze Inferno Wood Cracking Collapsing Loud Stressing_01.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
- `forest_wild_fire_sound_dataset/drive-download-20210504T145639Z-001/split file 3/videoplayback_01.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
- `forest_wild_fire_sound_dataset/drive-download-20210504T145639Z-001/split files 2/y2mate.com - DD Ambience  Building on Fire  Blaze Inferno Wood Cracking Collapsing Loud Stressing_1_01.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
Sample-rate counts from sampled files: {'44100': 250}; formats: {'wav': 289}

### Step 5 - Nature Subclass Mapping

- rain: NOT PRESENT
- sea_waves: NOT PRESENT
- wind: NOT PRESENT
- crackling_fire: folder/path regex ['crackl', '\\bfire\\b', 'wild[_ -]?fire', 'forest[_ -]?fire']; clips=289; estimated 5s clips=2878; sample_rates={'44100': 289}

### Step 7 - Problem Flags

- no metadata found; folder/path labels only
Decision: USE WITH CAUTION


## freefield1010

### Step 1 - Identify Dataset

Folder structure up to 3 levels deep:

- `01/`
- `02/`
- `03/`
- `04/`
- `05/`
- `06/`
- `07/`
- `08/`
- `09/`
- `10/`
- `metadataonly/`
- `  01/`
- `  02/`
- `  03/`
- `  04/`
- `  05/`
- `  06/`
- `  07/`
- `  08/`
- `  09/`
- `  10/`

Root non-audio files: None

Subfolder names: 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, metadataonly

### Step 2 - Count Audio Files

Total requested-extension audio files: 7690 (`wav/mp3/flac/ogg/m4a/aiff`). Including WebM/nonstandard: 7690.
Sample filenames:
- `freefield1010/06/109377.wav` parent `freefield1010/06`
- `freefield1010/03/75094.wav` parent `freefield1010/03`
- `freefield1010/08/101318.wav` parent `freefield1010/08`
- `freefield1010/06/39659.wav` parent `freefield1010/06`
- `freefield1010/08/117146.wav` parent `freefield1010/08`
Unique parent folder names containing requested audio: 01, 02, 03, 04, 05, 06, 07, 08, 09, 10

### Step 3 - Inspect Metadata Files

Metadata `freefield1010/metadataonly/md5sums.txt`
- Columns/keys: []
- Total row/key count: 11
- First 10 rows/lines:
```text
f7e912f2a343c825b0b34bfc1266f3df  01.zip
04f30793b32d9d395d1a748950c409d7  02.zip
67458a13698078e393e56016e56dd5ea  03.zip
11cc2b937fae71fe26da80de7d437e3c  04.zip
3f2bd0d21d1e60032378a41f8ee2d2a4  05.zip
f979114786df03fe418d0d715702c971  06.zip
e9be7edb08cee1514b5fa3e819619ab2  07.zip
ccc1cb66815a8e7f21dbbdb8bc219a12  08.zip
d9ec53bf5114bf882f145838e7125e18  09.zip
21e32d825e1c1d7eb93e42fd2c5cdf08  10.zip
```
Metadata `freefield1010/metadataonly/readme.txt`
- Columns/keys: []
- Total row/key count: 82
- First 10 rows/lines:
```text
freefield1010
=============

A dataset of standardised 10-second excerpts from Freesound field recordings.

Curated July 2013 by Dan Stowell at the Centre for Digital Music, QMUL, London.
Based on recordings from the Freesound archive <http://freesound.org/> hosted by the Music Technology Group, UPF, Barcelona.

Version 1.0

```
Metadata `freefield1010/01/100250.json`
- Columns/keys: ['similarity', 'channels', 'duration', 'samplerate', 'preview-hq-ogg', 'id', 'preview-lq-ogg', 'ref', 'analysis_stats', 'tags', 'serve', 'spectral_m', 'spectral_l', 'user', 'preview-hq-mp3', 'analysis_frames', 'license', 'url', 'type', 'preview-lq-mp3', 'original_filename', 'waveform_l', 'waveform_m', 'pack']
- Total row/key count: 24
- First 10 rows/lines:
```text
{'similarity': 'http://www.freesound.org/api/sounds/100250/similar/'}
{'channels': '2'}
{'duration': '340.986575964'}
{'samplerate': '44100.0'}
{'preview-hq-ogg': 'http://www.freesound.org/data/previews/100/100250_1578278-hq.ogg'}
{'id': '100250'}
{'preview-lq-ogg': 'http://www.freesound.org/data/previews/100/100250_1578278-lq.ogg'}
{'ref': 'http://www.freesound.org/api/sounds/100250/'}
{'analysis_stats': 'http://www.freesound.org/api/sounds/100250/analysis/'}
{'tags': "['birds', 'field-recording', 'gibbon', 'jungle', 'macaque', 'monkey', 'rainforest', 'wind', 'zoo']"}
```
Metadata `freefield1010/01/100275.json`
- Columns/keys: ['similarity', 'channels', 'duration', 'samplerate', 'preview-hq-ogg', 'id', 'preview-lq-ogg', 'ref', 'analysis_stats', 'tags', 'serve', 'spectral_m', 'spectral_l', 'user', 'preview-hq-mp3', 'analysis_frames', 'license', 'url', 'type', 'preview-lq-mp3', 'original_filename', 'waveform_l', 'waveform_m', 'pack']
- Total row/key count: 24
- First 10 rows/lines:
```text
{'similarity': 'http://www.freesound.org/api/sounds/100275/similar/'}
{'channels': '2'}
{'duration': '14.4951020408'}
{'samplerate': '44100.0'}
{'preview-hq-ogg': 'http://www.freesound.org/data/previews/100/100275_1578278-hq.ogg'}
{'id': '100275'}
{'preview-lq-ogg': 'http://www.freesound.org/data/previews/100/100275_1578278-lq.ogg'}
{'ref': 'http://www.freesound.org/api/sounds/100275/'}
{'analysis_stats': 'http://www.freesound.org/api/sounds/100275/analysis/'}
{'tags': "['ape', 'field-recording', 'jungle', 'monkey', 'primate', 'rainforest', 'spider-monkey', 'zoo']"}
```
Metadata `freefield1010/01/100276.json`
- Columns/keys: ['similarity', 'channels', 'duration', 'samplerate', 'preview-hq-ogg', 'id', 'preview-lq-ogg', 'ref', 'analysis_stats', 'tags', 'serve', 'spectral_m', 'spectral_l', 'user', 'preview-hq-mp3', 'analysis_frames', 'license', 'url', 'type', 'preview-lq-mp3', 'original_filename', 'waveform_l', 'waveform_m', 'pack']
- Total row/key count: 24
- First 10 rows/lines:
```text
{'similarity': 'http://www.freesound.org/api/sounds/100276/similar/'}
{'channels': '2'}
{'duration': '15.3'}
{'samplerate': '44100.0'}
{'preview-hq-ogg': 'http://www.freesound.org/data/previews/100/100276_1578278-hq.ogg'}
{'id': '100276'}
{'preview-lq-ogg': 'http://www.freesound.org/data/previews/100/100276_1578278-lq.ogg'}
{'ref': 'http://www.freesound.org/api/sounds/100276/'}
{'analysis_stats': 'http://www.freesound.org/api/sounds/100276/analysis/'}
{'tags': "['cage', 'chatter', 'field-recording', 'monkey', 'movement', 'primate', 'zoo']"}
```
Freefield1010 aggregated top JSON tags across per-clip metadata: {'field-recording': 15382, 'nature': 1854, 'birds': 1786, 'ambience': 1690, 'water': 1462, 'ambiance': 1198, 'city': 1150, 'voice': 1112, 'atmosphere': 874, 'rain': 848, 'train': 830, 'ambient': 812, 'traffic': 800, 'wind': 766, 'bird': 764, 'noise': 722, 'stereo': 706, 'people': 682, 'seoul': 656, 'korea': 654, 'street': 640, 'binaural': 630, 'thunder': 620, 'engine': 612, 'forest': 570, 'car': 536, 'waves': 520, 'spring': 488, 'storm': 450, 'footsteps': 438, 'cars': 432, 'voices': 426, 'river': 414, 'stream': 408, 'birdsong': 406, 'summer': 404, 'beach': 396, 'music': 396, 'crowd': 394, 'machine': 378, 'weather': 368, 'south-spain': 364, 'spanish': 362, 'night': 350, 'korean': 346, 'morning': 338, 'rotterdam': 334, 'sea': 326, 'station': 316, 'denmark': 300, 'insects': 300, 'soundscape': 298, 'spain': 278, 'xy': 268, 'sound': 268, 'thunderstorm': 266, 'truck': 260, 'scotland': 252, 'town': 250, 'park': 246, 'lightning': 242, 'speech': 238, 'outdoor': 236, 'background': 230, 'railway': 230, 'church': 226, 'subway': 224, 'metal': 224, 'ocean': 224, 'netherlands': 218, 'bus': 218, 'zoo': 214, 'drone': 210, 'industrial': 208, 'field': 208, 'crickets': 200, 'walking': 200, 'rumble': 198, 'winter': 198, 'waterfall': 196}

### Step 4 - Audio Quality Check

- `freefield1010/01/100250.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
- `freefield1010/02/101316.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
- `freefield1010/03/101317.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
Sample-rate counts from sampled files: {'44100': 250}; formats: {'wav': 7690}

### Step 5 - Nature Subclass Mapping

- rain: Freefield1010 per-clip JSON tags/original_filename regex ['\\brain\\b', '\\braining\\b', '\\brainfall\\b']; clips=434; estimated 5s clips=868; sample_rates={'44100': 434}
- sea_waves: Freefield1010 per-clip JSON tags/original_filename regex ['\\bsea\\b', '\\bocean\\b', '\\bwave\\b', '\\bwaves\\b', '\\bbeach\\b']; clips=400; estimated 5s clips=800; sample_rates={'44100': 400}
- wind: Freefield1010 per-clip JSON tags/original_filename regex ['\\bwind\\b', 'windy']; clips=428; estimated 5s clips=856; sample_rates={'44100': 428}
- crackling_fire: Freefield1010 per-clip JSON tags/original_filename regex ['crackl', '\\bfire\\b', '\\bflame\\b', '\\bburning\\b']; clips=70; estimated 5s clips=140; sample_rates={'44100': 70}

### Step 7 - Problem Flags

- None
Decision: USE


## urbansound8k

### Step 1 - Identify Dataset

Folder structure up to 3 levels deep:

- `fold1/`
- `fold10/`
- `fold2/`
- `fold3/`
- `fold4/`
- `fold5/`
- `fold6/`
- `fold7/`
- `fold8/`
- `fold9/`

Root non-audio files: urbansound8k/UrbanSound8K.csv

First 10 lines of `urbansound8k/UrbanSound8K.csv`:
```text
slice_file_name,fsID,start,end,salience,fold,classID,class
100032-3-0-0.wav,100032,0.0,0.317551,1,5,3,dog_bark
100263-2-0-117.wav,100263,58.5,62.5,1,5,2,children_playing
100263-2-0-121.wav,100263,60.5,64.5,1,5,2,children_playing
100263-2-0-126.wav,100263,63.0,67.0,1,5,2,children_playing
100263-2-0-137.wav,100263,68.5,72.5,1,5,2,children_playing
100263-2-0-143.wav,100263,71.5,75.5,1,5,2,children_playing
100263-2-0-161.wav,100263,80.5,84.5,1,5,2,children_playing
100263-2-0-3.wav,100263,1.5,5.5,1,5,2,children_playing
100263-2-0-36.wav,100263,18.0,22.0,1,5,2,children_playing
```
Subfolder names: fold1, fold10, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9

### Step 2 - Count Audio Files

Total requested-extension audio files: 8732 (`wav/mp3/flac/ogg/m4a/aiff`). Including WebM/nonstandard: 8732.
Sample filenames:
- `urbansound8k/fold8/70168-3-1-20.wav` parent `urbansound8k/fold8`
- `urbansound8k/fold4/61790-9-0-20.wav` parent `urbansound8k/fold4`
- `urbansound8k/fold9/79089-0-0-28.wav` parent `urbansound8k/fold9`
- `urbansound8k/fold6/162434-6-0-0.wav` parent `urbansound8k/fold6`
- `urbansound8k/fold2/34621-4-11-0.wav` parent `urbansound8k/fold2`
Unique parent folder names containing requested audio: fold1, fold10, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9

### Step 3 - Inspect Metadata Files

Metadata `urbansound8k/UrbanSound8K.csv`
- Columns/keys: ['slice_file_name', 'fsID', 'start', 'end', 'salience', 'fold', 'classID', 'class']
- Total row/key count: 8732
- Unique label-like values in `slice_file_name`: 100032-3-0-0.wav, 100263-2-0-117.wav, 100263-2-0-121.wav, 100263-2-0-126.wav, 100263-2-0-137.wav, 100263-2-0-143.wav, 100263-2-0-161.wav, 100263-2-0-3.wav, 100263-2-0-36.wav, 100648-1-0-0.wav, 100648-1-1-0.wav, 100648-1-2-0.wav, 100648-1-3-0.wav, 100648-1-4-0.wav, 100652-3-0-0.wav, 100652-3-0-1.wav, 100652-3-0-2.wav, 100652-3-0-3.wav, 100795-3-0-0.wav, 100795-3-1-0.wav, 100795-3-1-1.wav, 100795-3-1-2.wav, 100852-0-0-0.wav, 100852-0-0-1.wav, 100852-0-0-10.wav, 100852-0-0-11.wav, 100852-0-0-12.wav, 100852-0-0-13.wav, 100852-0-0-14.wav, 100852-0-0-15.wav, 100852-0-0-16.wav, 100852-0-0-17.wav, 100852-0-0-18.wav, 100852-0-0-19.wav, 100852-0-0-2.wav, 100852-0-0-20.wav, 100852-0-0-21.wav, 100852-0-0-22.wav, 100852-0-0-23.wav, 100852-0-0-24.wav, 100852-0-0-25.wav, 100852-0-0-26.wav, 100852-0-0-27.wav, 100852-0-0-28.wav, 100852-0-0-29.wav, 100852-0-0-3.wav, 100852-0-0-30.wav, 100852-0-0-4.wav, 100852-0-0-5.wav, 100852-0-0-6.wav, 100852-0-0-7.wav, 100852-0-0-8.wav, 100852-0-0-9.wav, 101281-3-0-0.wav, 101281-3-0-14.wav, 101281-3-0-5.wav, 101382-2-0-10.wav, 101382-2-0-12.wav, 101382-2-0-20.wav, 101382-2-0-21.wav, 101382-2-0-29.wav, 101382-2-0-33.wav, 101382-2-0-42.wav, 101382-2-0-45.wav, 101415-3-0-2.wav, 101415-3-0-3.wav, 101415-3-0-8.wav, 101729-0-0-1.wav, 101729-0-0-11.wav, 101729-0-0-12.wav, 101729-0-0-13.wav, 101729-0-0-14.wav, 101729-0-0-16.wav, 101729-0-0-17.wav, 101729-0-0-18.wav, 101729-0-0-19.wav, 101729-0-0-21.wav, 101729-0-0-22.wav, 101729-0-0-23.wav, 101729-0-0-24.wav, 101729-0-0-26.wav, 101729-0-0-28.wav, 101729-0-0-29.wav, 101729-0-0-3.wav, 101729-0-0-32.wav, 101729-0-0-33.wav, 101729-0-0-36.wav, 101729-0-0-37.wav, 101729-0-0-38.wav, 101729-0-0-39.wav, 101729-0-0-4.wav, 101729-0-0-40.wav, 101729-0-0-6.wav, 101729-0-0-9.wav, 101848-9-0-0.wav, 101848-9-0-1.wav, 101848-9-0-2.wav, 101848-9-0-3.wav, 101848-9-0-8.wav, 101848-9-0-9.wav, 102102-3-0-0.wav, 102103-3-0-0.wav, 102103-3-1-0.wav, 102104-3-0-0.wav, 102105-3-0-0.wav, 102106-3-0-0.wav, 102305-6-0-0.wav, 102547-3-0-2.wav, 102547-3-0-7.wav, 102547-3-0-8.wav, 102842-3-0-1.wav, 102842-3-1-0.wav, 102842-3-1-5.wav, 102842-3-1-6.wav, 102853-8-0-0.wav, 102853-8-0-1.wav, 102853-8-0-2.wav, 102853-8-0-3.wav, 102853-8-0-4.wav, 102853-8-0-5.wav ... (+180 more)
- Unique label-like values in `classID`: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Unique label-like values in `class`: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
- First 10 rows/lines:
```text
{'slice_file_name': '100032-3-0-0.wav', 'fsID': '100032', 'start': '0.0', 'end': '0.317551', 'salience': '1', 'fold': '5', 'classID': '3', 'class': 'dog_bark'}
{'slice_file_name': '100263-2-0-117.wav', 'fsID': '100263', 'start': '58.5', 'end': '62.5', 'salience': '1', 'fold': '5', 'classID': '2', 'class': 'children_playing'}
{'slice_file_name': '100263-2-0-121.wav', 'fsID': '100263', 'start': '60.5', 'end': '64.5', 'salience': '1', 'fold': '5', 'classID': '2', 'class': 'children_playing'}
{'slice_file_name': '100263-2-0-126.wav', 'fsID': '100263', 'start': '63.0', 'end': '67.0', 'salience': '1', 'fold': '5', 'classID': '2', 'class': 'children_playing'}
{'slice_file_name': '100263-2-0-137.wav', 'fsID': '100263', 'start': '68.5', 'end': '72.5', 'salience': '1', 'fold': '5', 'classID': '2', 'class': 'children_playing'}
{'slice_file_name': '100263-2-0-143.wav', 'fsID': '100263', 'start': '71.5', 'end': '75.5', 'salience': '1', 'fold': '5', 'classID': '2', 'class': 'children_playing'}
{'slice_file_name': '100263-2-0-161.wav', 'fsID': '100263', 'start': '80.5', 'end': '84.5', 'salience': '1', 'fold': '5', 'classID': '2', 'class': 'children_playing'}
{'slice_file_name': '100263-2-0-3.wav', 'fsID': '100263', 'start': '1.5', 'end': '5.5', 'salience': '1', 'fold': '5', 'classID': '2', 'class': 'children_playing'}
{'slice_file_name': '100263-2-0-36.wav', 'fsID': '100263', 'start': '18.0', 'end': '22.0', 'salience': '1', 'fold': '5', 'classID': '2', 'class': 'children_playing'}
{'slice_file_name': '100648-1-0-0.wav', 'fsID': '100648', 'start': '4.823402', 'end': '5.471927', 'salience': '2', 'fold': '10', 'classID': '1', 'class': 'car_horn'}
```
### Step 4 - Audio Quality Check

- `urbansound8k/fold1/101415-3-0-2.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
- `urbansound8k/fold10/100648-1-0-0.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
- `urbansound8k/fold2/100652-3-0-0.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
Sample-rate counts from sampled files: {'44100': 147, '48000': 24, '11025': 2, '24000': 1, '16000': 3, '96000': 2, '22050': 3}; formats: {'wav': 8732}

### Step 5 - Nature Subclass Mapping

- rain: NOT PRESENT
- sea_waves: NOT PRESENT
- wind: NOT PRESENT
- crackling_fire: NOT PRESENT

### Step 6 - Urban Subclass Mapping

- car_horn: UrbanSound8K.csv class == car_horn; clips=429; estimated 5s clips=0; sample_rates={'44100': 273, '16000': 21, '48000': 49}
- engine_idling: UrbanSound8K.csv class == engine_idling; clips=1000; estimated 5s clips=0; sample_rates={'44100': 461, '48000': 140}
- siren: UrbanSound8K.csv class == siren; clips=929; estimated 5s clips=0; sample_rates={'44100': 494, '48000': 217, '16000': 12, '11025': 9}
- jackhammer: UrbanSound8K.csv class == jackhammer; clips=1000; estimated 5s clips=0; sample_rates={'44100': 516, '96000': 12, '48000': 108}

### Step 7 - Problem Flags

- total useful 5s clips after filtering below 100
- sample rate below 16000 Hz observed
- clipping observed in sampled audio
Decision: SKIP


## 99Sounds Nature Sounds

### Step 1 - Identify Dataset

Folder structure up to 3 levels deep:

- `Animals/`
- `Forest/`
- `Rain/`
- `Water/`
- `Wind/`

Root non-audio files: 99Sounds Nature Sounds/ABOUT.txt

First 10 lines of `99Sounds Nature Sounds/ABOUT.txt`:
```text
Nature Sounds contains a set of royalty-free nature field recordings in the following categories: Animals, Forest, Rain, Water, and Wind.

It's an excellent addition to our most popular Rain Sounds and Water Sounds libraries. The new Nature Sounds offers more sonic variety and is ideal for sound design, film, social media, and music production.

Our friends from Free To Use Sounds recorded the included sounds in Taiwan and offered them as a free download to 99Sounds visitors.

You can find more field recordings from various places around the globe on the excellent Free To Use Sounds website.

The free nature sounds are provided in 24-bit 96 kHz stereo WAV format.

```
Subfolder names: Animals, Forest, Rain, Water, Wind

### Step 2 - Count Audio Files

Total requested-extension audio files: 83 (`wav/mp3/flac/ogg/m4a/aiff`). Including WebM/nonstandard: 83.
Sample filenames:
- `99Sounds Nature Sounds/Water/WATRSplsh-LR_Taiwan-Water, Splash, Textures, Creek, Alishan, 01.wav` parent `99Sounds Nature Sounds/Water`
- `99Sounds Nature Sounds/Animals/BIRDCrow-LR_Taiwan-Birds, Crow, Cawing, Park, Forest, Taipei, 02.wav` parent `99Sounds Nature Sounds/Animals`
- `99Sounds Nature Sounds/Water/WATRSplsh-LR_Taiwan-Water, Splash, Textures, Creek, Rooster Call, Alishan, 03.wav` parent `99Sounds Nature Sounds/Water`
- `99Sounds Nature Sounds/Rain/RAINMisc-LR_Thailand-Rain, Forest, Leaves, Bush, Tree, Wooden Roof, Cicadas, Splash, 06.wav` parent `99Sounds Nature Sounds/Rain`
- `99Sounds Nature Sounds/Animals/ANMLRdnt-LR_Taiwan, Animal, Rodent, Squirrel, Squeaks, Barks, Grunts, Water, Flow, Taipei, 06.wav` parent `99Sounds Nature Sounds/Animals`
Unique parent folder names containing requested audio: Animals, Forest, Rain, Water, Wind

### Step 3 - Inspect Metadata Files

Metadata `99Sounds Nature Sounds/ABOUT.txt`
- Columns/keys: []
- Total row/key count: 11
- First 10 rows/lines:
```text
Nature Sounds contains a set of royalty-free nature field recordings in the following categories: Animals, Forest, Rain, Water, and Wind.

It's an excellent addition to our most popular Rain Sounds and Water Sounds libraries. The new Nature Sounds offers more sonic variety and is ideal for sound design, film, social media, and music production.

Our friends from Free To Use Sounds recorded the included sounds in Taiwan and offered them as a free download to 99Sounds visitors.

You can find more field recordings from various places around the globe on the excellent Free To Use Sounds website.

The free nature sounds are provided in 24-bit 96 kHz stereo WAV format.

```
### Step 4 - Audio Quality Check

- `99Sounds Nature Sounds/Animals/ANMLAmph-LR_Taiwan-Animals, Amphibian, Frogs, Croaking, Vocalization, Village, Taiwan, 01.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
- `99Sounds Nature Sounds/Forest/AMBForst-LR_Thailand-Ambience, Forest, Night, Cicadas, River, Flow, Serene, 03.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
- `99Sounds Nature Sounds/Rain/RAINMisc-LR_Thailand-Rain, Forest, Leaves, Bush, Tree, Wooden Roof, Cicadas, Splash, 01.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
Sample-rate counts from sampled files: {'96000': 8, '192000': 17}; formats: {'wav': 83}

### Step 5 - Nature Subclass Mapping

- rain: folder/path strict selector; clips=6; estimated 5s clips=108; sample_rates={'96000': 6}
- sea_waves: folder/path strict selector; clips=4; estimated 5s clips=63; sample_rates={'192000': 4}
- wind: folder/path strict selector; clips=14; estimated 5s clips=174; sample_rates={'192000': 9}
- crackling_fire: NOT PRESENT

### Step 7 - Problem Flags

- None
Decision: USE


## 4060432

### Step 1 - Identify Dataset

Folder structure up to 3 levels deep:

- `FSD50K.dev_audio/`
- `FSD50K.doc/`
- `FSD50K.ground_truth/`
- `FSD50K.metadata/`
- `  collection/`

Root non-audio files: 4060432/FSD50K.dev_audio.z01, 4060432/FSD50K.dev_audio.z02, 4060432/FSD50K.dev_audio.z03, 4060432/FSD50K.dev_audio.z04, 4060432/FSD50K.dev_audio.z05, 4060432/FSD50K.dev_audio.zip, 4060432/FSD50K.doc.zip, 4060432/FSD50K.eval_audio.z01, 4060432/FSD50K.eval_audio.zip, 4060432/FSD50K.ground_truth.zip, 4060432/FSD50K.metadata.zip, 4060432/FSD50K_merged.zip

Subfolder names: FSD50K.dev_audio, FSD50K.doc, FSD50K.ground_truth, FSD50K.metadata

### Step 2 - Count Audio Files

Total requested-extension audio files: 40966 (`wav/mp3/flac/ogg/m4a/aiff`). Including WebM/nonstandard: 40966.
Sample filenames:
- `4060432/FSD50K.dev_audio/401619.wav` parent `4060432/FSD50K.dev_audio`
- `4060432/FSD50K.dev_audio/261091.wav` parent `4060432/FSD50K.dev_audio`
- `4060432/FSD50K.dev_audio/50405.wav` parent `4060432/FSD50K.dev_audio`
- `4060432/FSD50K.dev_audio/346519.wav` parent `4060432/FSD50K.dev_audio`
- `4060432/FSD50K.dev_audio/187266.wav` parent `4060432/FSD50K.dev_audio`
Unique parent folder names containing requested audio: FSD50K.dev_audio

### Step 3 - Inspect Metadata Files

Metadata `4060432/FSD50K.doc/README.md`
- Columns/keys: []
- Total row/key count: 202
- First 10 rows/lines:
```text

# Freesound Dataset 50k (FSD50K)

## Citation

If you use the FSD50K dataset, or part of it, please cite our paper:

>Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra. "FSD50K: an Open Dataset of Human-Labeled Sound Events", arXiv 2020.

### Data curators
```
Metadata `4060432/FSD50K.ground_truth/dev.csv`
- Columns/keys: ['fname', 'labels', 'mids', 'split']
- Total row/key count: 40966
- Unique label-like values in `fname`: 10000, 100005, 100006, 100007, 10001, 100011, 100017, 100018, 100019, 10002, 100021, 10003, 100035, 100036, 100039, 10004, 100041, 100042, 100043, 100047, 10005, 100069, 10007, 100072, 100078, 10008, 100080, 100081, 100082, 100083, 100092, 100093, 100095, 100096, 100098, 100099, 100101, 100104, 100114, 100124, 100143, 100144, 100150, 100155, 100183, 100207, 100219, 100246, 100248, 100249, 100279, 100300, 100303, 100304, 100305, 100306, 100307, 100308, 100309, 100310, 100311, 100312, 100313, 100315, 100316, 100317, 100318, 100319, 100321, 100322, 100323, 100324, 100325, 100326, 100327, 100328, 100329, 100330, 100331, 100332, 100333, 100334, 100335, 100337, 100338, 100357, 100364, 100390, 100400, 100455, 100457, 100459, 100460, 100461, 100462, 100465, 100476, 100632, 100633, 100634, 100636, 100637, 100638, 100640, 100641, 100642, 100643, 100644, 100646, 100681, 100733, 100738, 100762, 100767, 100771, 100772, 100795, 100800, 100849, 100850 ... (+180 more)
- Unique label-like values in `labels`: Accelerating_and_revving_and_vroom,Bird_vocalization_and_bird_call_and_bird_song,Car,Bird,Wild_animals,Animal,Motor_vehicle_(road),Vehicle,Engine, Accelerating_and_revving_and_vroom,Car,Motor_vehicle_(road),Vehicle,Engine, Accelerating_and_revving_and_vroom,Car,Vehicle,Motor_vehicle_(road),Engine, Accelerating_and_revving_and_vroom,Engine, Accelerating_and_revving_and_vroom,Engine,Mechanisms, Accelerating_and_revving_and_vroom,Motor_vehicle_(road),Vehicle,Engine, Accelerating_and_revving_and_vroom,Truck,Motor_vehicle_(road),Vehicle,Engine, Accelerating_and_revving_and_vroom,Vehicle,Engine, Accordion,Musical_instrument,Music, Acoustic_guitar,Drum,Human_voice,Percussion,Musical_instrument,Music,Guitar,Plucked_string_instrument, Acoustic_guitar,Guitar,Human_voice,Plucked_string_instrument,Musical_instrument,Music, Acoustic_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music, Acoustic_guitar,Human_voice,Guitar,Plucked_string_instrument,Musical_instrument,Music, Acoustic_guitar,Percussion,Human_voice,Musical_instrument,Music,Guitar,Plucked_string_instrument, Acoustic_guitar,Piano,Human_voice,Keyboard_(musical),Musical_instrument,Music,Guitar,Plucked_string_instrument, Acoustic_guitar,Plucked_string_instrument,Guitar,Musical_instrument,Music, Acoustic_guitar,Strum,Guitar,Plucked_string_instrument,Musical_instrument,Music, Aircraft,Vehicle, Alarm, Alarm,Bell, Alarm,Boat_and_Water_vehicle,Vehicle, Alarm,Car,Motor_vehicle_(road),Vehicle, Alarm,Car,Vehicle,Motor_vehicle_(road), Alarm,Clock,Mechanisms, Alarm,Dog,Domestic_animals_and_pets,Animal, Alarm,Door,Domestic_sounds_and_home_sounds, Alarm,Door,Train,Vehicle,Domestic_sounds_and_home_sounds,Rail_transport, Alarm,Engine,Train,Rail_transport,Vehicle, Alarm,Fireworks,Explosion, Alarm,Human_voice, Alarm,Insect,Wild_animals,Animal, Alarm,Keyboard_(musical),Musical_instrument,Music, Alarm,Ocean,Boat_and_Water_vehicle,Water,Vehicle, Alarm,Truck,Motor_vehicle_(road),Vehicle, Animal, Animal,Brass_instrument,Musical_instrument,Music, Animal,Human_voice, Applause,Chatter,Crowd,Clapping,Keyboard_(musical),Musical_instrument,Music,Human_group_actions,Hands, Applause,Cheering,Clapping,Human_group_actions,Hands, Applause,Cheering,Clapping,Shout,Human_group_actions,Human_voice,Hands, Applause,Cheering,Human_group_actions, Applause,Cheering,Human_voice,Human_group_actions, Applause,Cheering,Shout,Human_group_actions,Human_voice, Applause,Clapping,Human_group_actions,Hands, Applause,Clapping,Human_voice,Human_group_actions,Hands, Applause,Clapping,Speech,Human_voice,Human_group_actions,Hands, Applause,Crowd,Cheering,Human_group_actions, Applause,Crowd,Cheering,Human_voice,Human_group_actions, Applause,Crowd,Human_group_actions, Applause,Crowd,Laughter,Human_group_actions,Human_voice, Applause,Dishes_and_pots_and_pans,Chatter,Crowd,Cheering,Clapping,Child_speech_and_kid_speaking,Human_group_actions,Domestic_sounds_and_home_sounds,Hands,Speech,Human_voice, Applause,Hands,Human_group_actions, Applause,Harmonica,Cheering,Human_group_actions,Musical_instrument,Music, Applause,Harmonica,Human_group_actions,Musical_instrument,Music, Applause,Human_group_actions, Applause,Human_voice,Human_group_actions, Applause,Keyboard_(musical),Musical_instrument,Music,Human_group_actions, Applause,Knock,Chatter,Laughter,Human_group_actions,Human_voice,Door,Domestic_sounds_and_home_sounds, Applause,Laughter,Human_group_actions,Human_voice, Applause,Screaming,Crowd,Cheering,Clapping,Shout,Human_group_actions,Human_voice,Hands, Applause,Screaming,Crowd,Human_group_actions,Human_voice, Applause,Screaming,Guitar,Plucked_string_instrument,Musical_instrument,Music,Human_group_actions,Human_voice, Applause,Trumpet,Crowd,Bell,Human_group_actions,Brass_instrument,Musical_instrument,Music, Bark,Accelerating_and_revving_and_vroom,Car,Dog,Domestic_animals_and_pets,Animal,Motor_vehicle_(road),Vehicle,Engine, Bark,Animal,Dog,Domestic_animals_and_pets, Bark,Bird_vocalization_and_bird_call_and_bird_song,Dog,Domestic_animals_and_pets,Animal,Bird,Wild_animals, Bark,Chatter,Chirp_and_tweet,Livestock_and_farm_animals_and_working_animals,Dog,Domestic_animals_and_pets,Animal,Human_group_actions,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals, Bark,Chirp_and_tweet,Dog,Domestic_animals_and_pets,Animal,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals, Bark,Cricket,Buzz,Insect,Dog,Domestic_animals_and_pets,Animal,Wild_animals, Bark,Cricket,Dog,Domestic_animals_and_pets,Animal,Insect,Wild_animals, Bark,Dog,Domestic_animals_and_pets,Animal, Bark,Dog,Rain,Domestic_animals_and_pets,Animal,Water, Bark,Fill_(with_liquid),Trickle_and_dribble,Dog,Domestic_animals_and_pets,Animal,Liquid,Pour, Bark,Growling,Crow,Dog,Domestic_animals_and_pets,Animal,Bird,Wild_animals, Bark,Growling,Dog,Domestic_animals_and_pets,Animal, Bark,Livestock_and_farm_animals_and_working_animals,Dog,Domestic_animals_and_pets,Animal, Bark,Motor_vehicle_(road),Siren,Vehicle,Dog,Domestic_animals_and_pets,Animal,Alarm, Bark,Walk_and_footsteps,Chirp_and_tweet,Dog,Domestic_animals_and_pets,Animal,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals, Bark,Walk_and_footsteps,Dog,Domestic_animals_and_pets,Animal, Bass_drum,Cowbell,Snare_drum,Drum_kit,Drum,Percussion,Musical_instrument,Music,Bell, Bass_drum,Drum,Percussion,Musical_instrument,Music, Bass_drum,Drum_kit,Percussion,Musical_instrument,Music,Drum, Bass_drum,Keyboard_(musical),Musical_instrument,Music,Drum,Percussion, Bass_drum,Musical_instrument,Drum,Percussion,Music, Bass_drum,Snare_drum,Drum,Percussion,Musical_instrument,Music, Bass_drum,Snare_drum,Drum_kit,Drum,Percussion,Musical_instrument,Music, Bass_drum,Snare_drum,Piano,Drum,Percussion,Musical_instrument,Music,Keyboard_(musical), Bass_drum,Thump_and_thud,Drum,Percussion,Musical_instrument,Music, Bass_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music, Bass_guitar,Piano,Guitar,Plucked_string_instrument,Musical_instrument,Music,Keyboard_(musical), Bathtub_(filling_or_washing),Cat,Domestic_animals_and_pets,Animal,Domestic_sounds_and_home_sounds, Bathtub_(filling_or_washing),Child_speech_and_kid_speaking,Speech,Human_voice,Domestic_sounds_and_home_sounds, Bathtub_(filling_or_washing),Domestic_sounds_and_home_sounds, Bathtub_(filling_or_washing),Gurgling,Domestic_sounds_and_home_sounds,Water, Bathtub_(filling_or_washing),Liquid,Domestic_sounds_and_home_sounds, Bathtub_(filling_or_washing),Sink_(filling_or_washing),Domestic_sounds_and_home_sounds, Bathtub_(filling_or_washing),Sink_(filling_or_washing),Water_tap_and_faucet,Domestic_sounds_and_home_sounds, Bathtub_(filling_or_washing),Splash_and_splatter,Liquid,Domestic_sounds_and_home_sounds, Bathtub_(filling_or_washing),Water,Domestic_sounds_and_home_sounds, Bathtub_(filling_or_washing),Water,Liquid,Domestic_sounds_and_home_sounds, Bathtub_(filling_or_washing),Water,Splash_and_splatter,Liquid,Domestic_sounds_and_home_sounds, Bathtub_(filling_or_washing),Water_tap_and_faucet,Domestic_sounds_and_home_sounds, Bell, Bell,Doorbell,Door,Domestic_sounds_and_home_sounds,Alarm, Bell,Drum,Drum_kit,Percussion,Musical_instrument,Music, Bell,Livestock_and_farm_animals_and_working_animals,Animal, Bell,Mechanisms, Bell,Percussion,Musical_instrument,Music, Bell,Telephone,Alarm, Bell,Train,Rail_transport,Vehicle, Bicycle,Vehicle, Bicycle_bell,Alarm,Bicycle,Vehicle,Bell, Bicycle_bell,Bell,Alarm,Bicycle,Vehicle, Bicycle_bell,Bell,Bicycle,Vehicle,Alarm, Bicycle_bell,Bicycle,Vehicle,Alarm,Bell, Bird,Bell,Wild_animals,Animal, Bird,Breathing,Wild_animals,Animal,Respiratory_sounds, Bird,Chicken_and_rooster,Wild_animals,Animal,Fowl,Livestock_and_farm_animals_and_working_animals, Bird,Dog,Wild_animals,Animal,Domestic_animals_and_pets, Bird,Fireworks,Explosion,Wild_animals,Animal ... (+180 more)
- First 10 rows/lines:
```text
{'fname': '64760', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf', 'split': 'train'}
{'fname': '16399', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf', 'split': 'train'}
{'fname': '16401', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf', 'split': 'train'}
{'fname': '16402', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf', 'split': 'train'}
{'fname': '16404', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf', 'split': 'train'}
{'fname': '345111', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf', 'split': 'val'}
{'fname': '64761', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf', 'split': 'train'}
{'fname': '268259', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf', 'split': 'train'}
{'fname': '64762', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf', 'split': 'train'}
{'fname': '160826', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf', 'split': 'val'}
```
Metadata `4060432/FSD50K.ground_truth/eval.csv`
- Columns/keys: ['fname', 'labels', 'mids']
- Total row/key count: 10231
- Unique label-like values in `fname`: 100, 100030, 100032, 1001, 100210, 100265, 100355, 100426, 1005, 100647, 1007, 100783, 100785, 100786, 100791, 1008, 100839, 100884, 100895, 100897, 1009, 100904, 1010, 1011, 101246, 1013, 101354, 101355, 1014, 101468, 101471, 101480, 101489, 101509, 1017, 101732, 101733, 101734, 101744, 101746, 101762, 101765, 101793, 101794, 101796, 101797, 101807, 101809, 101811, 101823, 101827, 101843, 101846, 101851, 101881, 101933, 101934, 101948, 102308, 102422, 102437, 102542, 102547, 102548, 102562, 102563, 102568, 102684, 102715, 102780, 102789, 1028, 102908, 102952, 102953, 102964, 102965, 102968, 102993, 102997, 103028, 103029, 103031, 103033, 103073, 103076, 1032, 103333, 103355, 103373, 103576, 103650, 1037, 104110, 1042, 104209, 104210, 104212, 104214, 104218, 104222, 104224, 104225, 104228, 104229, 104231, 104235, 104243, 104244, 104248, 104251, 104275, 104276, 104277, 104293, 1043, 104305, 1044, 104420, 104425 ... (+180 more)
- Unique label-like values in `labels`: Accelerating_and_revving_and_vroom,Car,Motor_vehicle_(road),Vehicle,Engine, Accelerating_and_revving_and_vroom,Motor_vehicle_(road),Vehicle,Engine, Accelerating_and_revving_and_vroom,Truck,Motor_vehicle_(road),Vehicle,Engine, Accordion,Conversation,Musical_instrument,Music,Speech,Human_voice, Accordion,Drum,Percussion,Musical_instrument,Music, Accordion,Finger_snapping,Drum,Percussion,Musical_instrument,Music,Hands, Accordion,Glockenspiel,Musical_instrument,Music,Mallet_percussion,Percussion, Accordion,Musical_instrument,Music, Accordion,Plucked_string_instrument,Musical_instrument,Music, Accordion,Tambourine,Singing,Bowed_string_instrument,Musical_instrument,Human_voice,Music,Percussion, Accordion,Tambourine,Singing,Musical_instrument,Human_voice,Music,Percussion, Accordion,Wind_instrument_and_woodwind_instrument,Musical_instrument,Music, Acoustic_guitar,Accordion,Guitar,Plucked_string_instrument,Musical_instrument,Music, Acoustic_guitar,Bass_drum,Tambourine,Guitar,Plucked_string_instrument,Musical_instrument,Music,Drum,Percussion, Acoustic_guitar,Female_singing,Laughter,Human_voice,Guitar,Plucked_string_instrument,Musical_instrument,Music,Singing, Acoustic_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music, Acoustic_guitar,Keyboard_(musical),Musical_instrument,Music,Guitar,Plucked_string_instrument, Acoustic_guitar,Rattle_(instrument),Guitar,Plucked_string_instrument,Musical_instrument,Music,Percussion, Acoustic_guitar,Strum,Guitar,Plucked_string_instrument,Musical_instrument,Music, Acoustic_guitar,Tambourine,Guitar,Plucked_string_instrument,Musical_instrument,Music,Percussion, Acoustic_guitar,Wind_instrument_and_woodwind_instrument,Musical_instrument,Music,Guitar,Plucked_string_instrument, Acoustic_guitar,Yell,Strum,Guitar,Plucked_string_instrument,Musical_instrument,Music,Shout,Human_voice, Aircraft,Vehicle, Alarm, Alarm,Bell, Alarm,Car,Motor_vehicle_(road),Vehicle, Alarm,Engine, Alarm,Engine,Aircraft,Vehicle, Alarm,Engine,Mechanisms,Tools, Alarm,Engine,Train,Music,Rail_transport,Vehicle, Alarm,Fireworks,Car,Explosion,Motor_vehicle_(road),Vehicle, Alarm,Keyboard_(musical),Musical_instrument,Music, Alarm,Keyboard_(musical),Truck,Train,Musical_instrument,Music,Motor_vehicle_(road),Vehicle,Rail_transport, Alarm,Mechanisms, Alarm,Train,Rail_transport,Vehicle, Alarm,Truck,Boat_and_Water_vehicle,Motor_vehicle_(road),Vehicle, Alarm,Truck,Motor_vehicle_(road),Vehicle, Alarm,Truck,Ocean,Boat_and_Water_vehicle,Motor_vehicle_(road),Vehicle,Water, Applause,Chatter,Cheering,Clapping,Guitar,Shout,Plucked_string_instrument,Musical_instrument,Music,Human_group_actions,Human_voice,Hands, Applause,Chatter,Cheering,Clapping,Human_group_actions,Hands, Applause,Chatter,Cheering,Clapping,Shout,Human_group_actions,Human_voice,Hands, Applause,Chatter,Clapping,Human_group_actions,Hands, Applause,Chatter,Clapping,Shout,Human_group_actions,Human_voice,Hands, Applause,Chatter,Crowd,Cheering,Clapping,Human_group_actions,Hands, Applause,Chatter,Crowd,Cheering,Clapping,Shout,Human_group_actions,Human_voice,Hands, Applause,Chatter,Crowd,Clapping,Child_speech_and_kid_speaking,Laughter,Human_group_actions,Human_voice,Hands,Speech, Applause,Chatter,Crowd,Clapping,Human_voice,Human_group_actions,Hands, Applause,Chatter,Crowd,Clapping,Laughter,Human_group_actions,Human_voice,Hands, Applause,Chatter,Crowd,Clapping,Shout,Human_group_actions,Human_voice,Hands, Applause,Cheering,Clapping,Drum,Percussion,Musical_instrument,Music,Human_group_actions,Hands, Applause,Cheering,Clapping,Human_group_actions,Hands, Applause,Cheering,Clapping,Laughter,Human_group_actions,Human_voice,Hands, Applause,Cheering,Clapping,Shout,Human_group_actions,Human_voice,Hands, Applause,Cheering,Clapping,Speech,Music,Human_voice,Human_group_actions,Hands, Applause,Cheering,Human_group_actions, Applause,Cheering,Shout,Human_group_actions,Human_voice, Applause,Clapping,Cough,Human_group_actions,Hands,Respiratory_sounds, Applause,Clapping,Engine,Laughter,Aircraft,Human_group_actions,Human_voice,Hands,Vehicle, Applause,Clapping,Human_group_actions,Hands, Applause,Clapping,Motor_vehicle_(road),Siren,Shout,Vehicle,Human_group_actions,Alarm,Human_voice,Hands, Applause,Clapping,Shout,Human_group_actions,Human_voice,Hands, Applause,Clapping,Shout,Thump_and_thud,Human_group_actions,Human_voice,Hands, Applause,Clapping,Speech,Human_voice,Human_group_actions,Hands, Applause,Crowd,Car_passing_by,Race_car_and_auto_racing,Engine,Human_group_actions,Car,Motor_vehicle_(road),Vehicle, Applause,Crowd,Cheering,Clapping,Drum,Shout,Percussion,Musical_instrument,Music,Human_group_actions,Human_voice,Hands, Applause,Crowd,Cheering,Clapping,Human_group_actions,Hands, Applause,Crowd,Cheering,Clapping,Laughter,Shout,Human_group_actions,Human_voice,Hands, Applause,Crowd,Cheering,Clapping,Shout,Human_group_actions,Human_voice,Hands, Applause,Crowd,Cheering,Clapping,Shout,Music,Human_group_actions,Human_voice,Hands, Applause,Crowd,Cheering,Human_group_actions, Applause,Crowd,Clapping,Human_group_actions,Hands, Applause,Crowd,Clapping,Laughter,Human_group_actions,Human_voice,Hands, Applause,Crowd,Human_group_actions, Applause,Crowd,Laughter,Human_group_actions,Human_voice, Applause,Dishes_and_pots_and_pans,Crowd,Cheering,Clapping,Shout,Human_group_actions,Domestic_sounds_and_home_sounds,Human_voice,Hands, Applause,Female_singing,Male_singing,Keyboard_(musical),Bell,Drum,Drum_kit,Musical_instrument,Music,Percussion,Human_group_actions,Singing,Human_voice, Applause,Human_group_actions, Applause,Knock,Cheering,Clapping,Laughter,Human_group_actions,Human_voice,Hands,Door,Domestic_sounds_and_home_sounds, Applause,Screaming,Chatter,Crowd,Cheering,Clapping,Human_group_actions,Human_voice,Hands, Applause,Screaming,Chatter,Crowd,Cheering,Clapping,Shout,Human_group_actions,Human_voice,Hands, Applause,Screaming,Crowd,Cheering,Yell,Clapping,Human_group_actions,Human_voice,Shout,Hands, Applause,Singing,Organ,Human_group_actions,Human_voice,Keyboard_(musical),Musical_instrument,Music, Applause,Tap,Clapping,Human_group_actions,Hands, Applause,Thunder,Chatter,Cheering,Human_group_actions,Thunderstorm, Bark,Bird_vocalization_and_bird_call_and_bird_song,Dog,Domestic_animals_and_pets,Animal,Bird,Wild_animals, Bark,Car_passing_by,Dog,Domestic_animals_and_pets,Animal,Car,Motor_vehicle_(road),Vehicle, Bark,Chatter,Chirp_and_tweet,Dog,Domestic_animals_and_pets,Animal,Human_group_actions,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals, Bark,Chirp_and_tweet,Accelerating_and_revving_and_vroom,Car,Dog,Domestic_animals_and_pets,Animal,Motor_vehicle_(road),Vehicle,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals,Engine, Bark,Chirp_and_tweet,Chicken_and_rooster,Dog,Domestic_animals_and_pets,Animal,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals,Fowl,Livestock_and_farm_animals_and_working_animals, Bark,Chirp_and_tweet,Chink_and_clink,Wind,Dog,Domestic_animals_and_pets,Animal,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals,Glass, Bark,Chirp_and_tweet,Dog,Domestic_animals_and_pets,Animal,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals, Bark,Chirp_and_tweet,Growling,Dog,Domestic_animals_and_pets,Animal,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals, Bark,Chirp_and_tweet,Livestock_and_farm_animals_and_working_animals,Dog,Domestic_animals_and_pets,Animal,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals, Bark,Chirp_and_tweet,Speech,Human_voice,Dog,Domestic_animals_and_pets,Animal,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals, Bark,Chirp_and_tweet,Traffic_noise_and_roadway_noise,Vehicle_horn_and_car_horn_and_honking,Alarm,Car,Motor_vehicle_(road),Vehicle,Dog,Domestic_animals_and_pets,Animal,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals, Bark,Church_bell,Cricket,Dog,Domestic_animals_and_pets,Animal,Bell,Insect,Wild_animals, Bark,Cowbell,Dog,Domestic_animals_and_pets,Animal,Bell, Bark,Cricket,Dog,Domestic_animals_and_pets,Animal,Insect,Wild_animals, Bark,Dog,Domestic_animals_and_pets,Animal, Bark,Drum,Dog,Domestic_animals_and_pets,Animal,Percussion,Musical_instrument,Music, Bark,Engine,Dog,Domestic_animals_and_pets,Animal, Bark,Engine,Fireworks,Human_voice,Car,Dog,Domestic_animals_and_pets,Animal,Explosion,Motor_vehicle_(road),Vehicle, Bark,Fixed-wing_aircraft_and_airplane,Engine,Dog,Domestic_animals_and_pets,Animal,Aircraft,Vehicle, Bark,Frying_(food),Dog,Domestic_animals_and_pets,Animal,Domestic_sounds_and_home_sounds, Bark,Growling,Bird_vocalization_and_bird_call_and_bird_song,Dog,Domestic_animals_and_pets,Animal,Bird,Wild_animals, Bark,Growling,Dog,Domestic_animals_and_pets,Animal, Bark,Growling,Dog,Domestic_animals_and_pets,Animal,Wild_animals, Bark,Growling,Speech,Human_voice,Dog,Domestic_animals_and_pets,Animal, Bark,Human_voice,Dog,Domestic_animals_and_pets,Animal, Bark,Keys_jangling,Dog,Domestic_animals_and_pets,Animal,Domestic_sounds_and_home_sounds, Bark,Meow,Dog,Domestic_animals_and_pets,Animal,Cat, Bark,Rain,Fireworks,Dog,Domestic_animals_and_pets,Animal,Water,Explosion, Bark,Raindrop,Traffic_noise_and_roadway_noise,Rain,Dog,Domestic_animals_and_pets,Animal,Water,Motor_vehicle_(road),Vehicle, Bark,Screaming,Run,Dog,Domestic_animals_and_pets,Animal,Human_voice, Bark,Snare_drum,Livestock_and_farm_animals_and_working_animals,Dog,Domestic_animals_and_pets,Animal,Drum,Percussion,Musical_instrument,Music, Bark,Speech,Human_voice,Dog,Domestic_animals_and_pets,Animal, Bark,Stream,Cricket,Wind,Splash_and_splatter,Dog,Domestic_animals_and_pets,Animal,Water,Liquid,Insect,Wild_animals, Bark,Thunder,Chirp_and_tweet,Rain,Dog,Domestic_animals_and_pets,Animal,Water,Thunderstorm,Bird_vocalization_and_bird_call_and_bird_song,Bird,Wild_animals, Bark,Traffic_noise_and_roadway_noise,Dog,Domestic_animals_and_pets,Animal,Motor_vehicle_(road),Vehicle, Bark,Water,Crying_and_sobbing,Dog,Domestic_animals_and_pets,Animal,Human_voice ... (+180 more)
- First 10 rows/lines:
```text
{'fname': '37199', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf'}
{'fname': '175151', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf'}
{'fname': '253463', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf'}
{'fname': '329838', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf'}
{'fname': '1277', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf'}
{'fname': '30149', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf'}
{'fname': '331398', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf'}
{'fname': '333246', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf'}
{'fname': '232924', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf'}
{'fname': '42378', 'labels': 'Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music', 'mids': '/m/02sgy,/m/0342h,/m/0fx80y,/m/04szw,/m/04rlf'}
```
Metadata `4060432/FSD50K.ground_truth/vocabulary.csv`
- Columns/keys: ['0', 'Accelerating_and_revving_and_vroom', '/m/07q2z82']
- Total row/key count: 199
- First 10 rows/lines:
```text
{'0': '1', 'Accelerating_and_revving_and_vroom': 'Accordion', '/m/07q2z82': '/m/0mkg'}
{'0': '2', 'Accelerating_and_revving_and_vroom': 'Acoustic_guitar', '/m/07q2z82': '/m/042v_gx'}
{'0': '3', 'Accelerating_and_revving_and_vroom': 'Aircraft', '/m/07q2z82': '/m/0k5j'}
{'0': '4', 'Accelerating_and_revving_and_vroom': 'Alarm', '/m/07q2z82': '/m/07pp_mv'}
{'0': '5', 'Accelerating_and_revving_and_vroom': 'Animal', '/m/07q2z82': '/m/0jbk'}
{'0': '6', 'Accelerating_and_revving_and_vroom': 'Applause', '/m/07q2z82': '/m/028ght'}
{'0': '7', 'Accelerating_and_revving_and_vroom': 'Bark', '/m/07q2z82': '/m/05tny_'}
{'0': '8', 'Accelerating_and_revving_and_vroom': 'Bass_drum', '/m/07q2z82': '/m/0bm02'}
{'0': '9', 'Accelerating_and_revving_and_vroom': 'Bass_guitar', '/m/07q2z82': '/m/018vs'}
{'0': '10', 'Accelerating_and_revving_and_vroom': 'Bathtub_(filling_or_washing)', '/m/07q2z82': '/m/03dnzn'}
```
Metadata `4060432/FSD50K.metadata/class_info_FSD50K.json`
- Columns/keys: ['/m/0dv3j', '/m/0brhx', '/m/0316dw', '/m/018vs', '/m/0bm02', '/m/0ngt1', '/m/0jb2l', '/m/02sgy', '/m/07pb8fc', '/m/07pjwq1', '/m/0g6b5', '/m/03m9d0z', '/m/025_jnm', '/m/07plz5l', '/m/0463cq4', '/m/07p6fty', '/m/014zdl', '/m/0242l', '/m/04rlf', '/m/07rqsjt', '/m/01b82r', '/m/0c2wf', '/m/0ch8v', '/m/01dwxx', '/m/0838f', '/m/07qrkrw', '/m/07rjwbb', '/m/081rb', '/m/01j3sz', '/m/01280g', '/m/02_nn', '/m/09l8g', '/m/07sr1lc', '/m/01yrx', '/m/0ltv', '/m/0k4j', '/m/012f08', '/m/07swgks', '/m/015p6', '/m/025rv6n', '/m/0jbk', '/m/02_41', '/m/034srq', '/m/02hnl', '/m/07rgt08', '/m/023pjk', '/m/0dxrf', '/m/05r5wn', '/m/07qqyl4', '/m/02mk9', '/m/0j45pbj', '/t/dd00134', '/m/0k5j', '/m/07gql', '/m/07r5v4s', '/m/07prgkl', '/t/dd00003', '/t/dd00004', '/m/07q6cd_', '/m/04s8yn', '/m/01lsmm', '/m/04brg2', '/m/0j6m2', '/m/07rn7sz', '/m/039jq', '/m/02rtxlg', '/m/07pbtc8', '/m/028ght', '/m/02yds9', '/m/02x984l', '/t/dd00112', '/m/0dv5r', '/m/09b5t', '/m/053hz1', '/m/05kq4', '/t/dd00130', '/m/01hgjl', '/m/068hy', '/m/06_fw', '/m/083vt', '/m/07rrlb6', '/m/0l15bq', '/m/0239kh', '/m/03vt0', '/m/06d_3', '/m/042v_gx', '/m/0dwsp', '/m/01m2v', '/m/0342h', '/m/07rjzl8', '/m/0ytgt', '/m/07pp_mv', '/m/01p970', '/m/085jw', '/m/07yv9', '/m/01s0vc', '/m/0gy1t2s', '/m/07qcpgn', '/m/0mkg', '/m/0mbct']
- Total row/key count: 200
- First 10 rows/lines:
```text
{'/m/0dv3j': '{\'faq\': \'<div class="ui accordion">\\r\\n\\r\\n class="title">\\r\\n  \\t<i class="dropdown icon"></i>\\r\\n  \\tAre frying, hissing or sizzling sounds considered ‘Boiling’ sounds?\\r\\n\\t</div>\\r\\n\\t class="content">\\r\\n  \\t<p>No.</p>\\r\\n\\t</div>\\r\\n\\r\\n class="title">\\r\\n  \\t<i class="dropdown icon"></i>\\r\\n  \\tAre ‘Boiling’ sounds containing hissing sounds from gas stoves or kettles considered <i>Present and predominant</i> sounds?\\r\\n\\t</div>\\r\\n\\t class="content">\\r\\n  \\t<p>No. When ‘Boiling’ and hissi'}
{'/m/0brhx': '{\'faq\': \'<div class="ui accordion">\\r\\n\\r\\n class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Are heavily processed sounds (e.g., distorted or with audio effects) considered ‘Speech synthesizer’ sounds?\\r\\n    </div>\\r\\n     class="content">\\r\\n      <p>Only consider them ‘Speech synthesizer’ if the sounds are easy to recognize.</p>\\r\\n    </div>\\r\\n\\r\\n\\r\\n</div>\', \'examples\': [256370, 256368, 256359], \'verification_examples\': [2832]}'}
{'/m/0316dw': '{\'faq\': \'<div class="ui accordion">\\r\\n     class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Are mouse clicks considered ‘Typing’ sounds?\\r\\n    </div>\\r\\n     class="content">\\r\\n      <p>No, only consider them ‘Typing’ if the sound is the result of the act of writing or inputting text. If mouse clicking sounds are mixed with \\\'Typing\\\' sounds, you should rate them as \\\'Present but not predominant\\\'.</p>\\r\\n    </div>\\r\\n\\r\\n     class="title">\\r\\n      <i class="dropdown icon"></'}
{'/m/018vs': '{\'faq\': \'<div class="ui accordion">\\r\\n class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Are atypical uses of the bass guitar considered ‘Bass guitar’ sounds?\\r\\n    </div>\\r\\n     class="content">\\r\\n      <p>Only consider them  ‘Bass guitar’ if the source is easy to recognize.</p>\\r\\n    </div>\\r\\n\\r\\n class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Any bass-lines are considered ‘Bass guitar’ sounds?\\r\\n    </div>\\r\\n     class="content">\\r\\n      <p>Not necessaril'}
{'/m/0bm02': '{\'faq\': \'<div class="ui accordion">\\r\\n    class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Are synthetic sounds considered ‘Bass drum’ sounds?\\r\\n    </div>\\r\\n     class="content">\\r\\n            <p>Bad replicas or drum machines should be rated as <i>Not present</i>.</p>\\r\\n    </div>\\r\\n</div>\', \'examples\': [38903, 78815], \'verification_examples\': [90623, 90621, 41420, 36004, 3166, 41148]}'}
{'/m/0ngt1': '{\'faq\': \'<div class="ui accordion">\\r\\n  \\r\\n class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Are synthetic thunders considered ‘Thunder’ sounds?\\r\\n    </div>\\r\\n     class="content">\\r\\n      <p>Only consider them ‘Thunder’ if they sound as real as an original one. Bad replicas should be rated as Not Present.</p>\\r\\n    </div>\\r\\n class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Are heavily processed sounds (e.g., distorted or with audio effects) considered ‘Thunde'}
{'/m/0jb2l': '{\'faq\': \'<div class="ui accordion">\\r\\n     class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Are rain or wind sounds considered ‘Thunderstorm’ sounds?\\r\\n    </div>\\r\\n     class="content">\\r\\n      <p>Not necessarily, only consider them ‘Thunderstorm’ if the storm has audible thunders. To help you decide, you can check the sounds’ metadata in Freesound by clicking the <img class="ui inline image" style="width: 80px;" src="/static/img/freesound_logo_color.png"> link.</p>\\r\\n    </d'}
{'/m/02sgy': '{\'faq\': \'<div class="ui accordion">\\r\\n\\r\\n            class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Are ‘Bass guitar’ sounds considered \\\'Electric guitar\\\' sounds?\\r\\n             </div>\\r\\n              class="content">\\r\\n      <p>No, be sure to distinguish \\\'Electric guitar\\\' sounds from ‘Bass guitar’. Understanding the siblings and direct children will help you distinguish \\\'Electric guitar\\\' from other similar categories. Click on any <span class="ui label">category</span>'}
{'/m/07pb8fc': '{\'faq\': \'<div class="ui accordion">\\r\\n            class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Are ‘Accelerating, revving, vroom\\\', \\\'Engine starting’ or shutting off sounds considered \\\'Idling\\\' sounds?\\r\\n             </div>\\r\\n              class="content">\\r\\n      <p>No, be sure to distinguish \\\'Idling\\\' sounds from them. Understanding the siblings and direct children will help you distinguish \\\'Idling\\\' from other similar categories. Click on any <span class="ui label">c'}
{'/m/07pjwq1': '{\'faq\': \'<div class="ui accordion">\\r\\n\\r\\n\\r\\n     class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Are human imitations considered ‘Buzz’ sounds?\\r\\n    </div>\\r\\n     class="content">\\r\\n      <p>Yes, as far as they phonetically imitate, resemble, or suggest the ‘Buzz’ sound (see the \\\'Onomatopoeia\\\' description).</p>\\r\\n    </div>\\r\\n\\r\\n  class="title">\\r\\n      <i class="dropdown icon"></i>\\r\\n      Are any vibration sounds like \\\'Twang\\\' or \\\'Ding\\\' considered \\\'Buzz\\\' sound'}
```
Metadata `4060432/FSD50K.metadata/collection/collection_dev.csv`
- Columns/keys: ['fname', 'labels', 'mids']
- Total row/key count: 40966
- Unique label-like values in `fname`: 10000, 100005, 100006, 100007, 10001, 100011, 100017, 100018, 100019, 10002, 100021, 10003, 100035, 100036, 100039, 10004, 100041, 100042, 100043, 100047, 10005, 100069, 10007, 100072, 100078, 10008, 100080, 100081, 100082, 100083, 100092, 100093, 100095, 100096, 100098, 100099, 100101, 100104, 100114, 100124, 100143, 100144, 100150, 100155, 100183, 100207, 100219, 100246, 100248, 100249, 100279, 100300, 100303, 100304, 100305, 100306, 100307, 100308, 100309, 100310, 100311, 100312, 100313, 100315, 100316, 100317, 100318, 100319, 100321, 100322, 100323, 100324, 100325, 100326, 100327, 100328, 100329, 100330, 100331, 100332, 100333, 100334, 100335, 100337, 100338, 100357, 100364, 100390, 100400, 100455, 100457, 100459, 100460, 100461, 100462, 100465, 100476, 100632, 100633, 100634, 100636, 100637, 100638, 100640, 100641, 100642, 100643, 100644, 100646, 100681, 100733, 100738, 100762, 100767, 100771, 100772, 100795, 100800, 100849, 100850 ... (+180 more)
- Unique label-like values in `labels`: Accelerating_and_revving_and_vroom, Accordion, Acoustic_guitar, Acoustic_guitar,Plucked_string_instrument, Acoustic_guitar,Strum, Air_conditioning, Air_conditioning,Fixed-wing_aircraft_and_airplane, Air_horn_and_truck_horn, Aircraft, Aircraft_engine, Aircraft_engine,Aircraft, Aircraft_engine,Fixed-wing_aircraft_and_airplane, Aircraft_engine,Fixed-wing_aircraft_and_airplane,Aircraft, Aircraft_engine,Idling, Alarm, Alarm,Alarm_clock, Alarm,Bus, Alarm,Car_alarm, Alarm,Ding-dong, Alarm,Door, Alarm,Door,Train,Vehicle, Alarm,Engine,Train, Alarm,Female_speech_and_woman_speaking,Walk_and_footsteps,Sliding_door, Alarm,Female_speech_and_woman_speaking,Walk_and_footsteps,Subway_and_metro_and_underground,Sliding_door, Alarm,Fireworks, Alarm,Human_voice, Alarm,Insect, Alarm,Microwave_oven, Alarm,Siren, Alarm,Slam, Alarm,Subway_and_metro_and_underground, Alarm,Telephone, Alarm,Truck, Alarm_clock, Alarm_clock,Tick-tock, Alto_saxophone, Alto_saxophone,Clarinet, Alto_saxophone,Saxophone, Ambulance_(siren), Ambulance_(siren),Police_car_(siren), Ambulance_(siren),Police_car_(siren),Helicopter, Ambulance_(siren),Vehicle, Animal, Animal,Bark, Animal,Bark,Dog, Animal,Bark,Howl, Animal,Bark,Howl,Dog, Animal,Bird,Caw, Animal,Bird_vocalization_and_bird_call_and_bird_song, Animal,Cattle_and_bovinae, Animal,Dog,Growling, Animal,French_horn, Animal,Howl, Animal,Human_voice, Animal,Meow, Animal,Oink, Animal,Pig, Animal,Pig,Oink, Animal,Screaming, Applause, Applause,Cheering, Applause,Cheering,Clapping, Applause,Cheering,Shout,Clapping, Applause,Clapping, Applause,Crowd, Applause,Crowd,Cheering, Applause,Crowd,Human_voice,Cheering, Applause,Crowd,Laughter, Applause,Dishes_and_pots_and_pans,Chatter,Crowd,Cheering,Clapping,Child_speech_and_kid_speaking, Applause,Harmonica, Applause,Harmonica,Cheering, Applause,Human_voice, Applause,Human_voice,Cheering, Applause,Human_voice,Clapping, Applause,Knock,Chatter, Applause,Laughter, Applause,Screaming,Crowd, Applause,Screaming,Crowd,Cheering,Clapping, Applause,Whoop,Cheering, Artillery_fire, Artillery_fire,Boom, Babbling, Babbling,Baby_cry_and_infant_cry, Babbling,Bathtub_(filling_or_washing),Child_speech_and_kid_speaking, Babbling,Child_speech_and_kid_speaking, Babbling,Human_voice, Baby_cry_and_infant_cry, Baby_cry_and_infant_cry,Child_speech_and_kid_speaking, Baby_cry_and_infant_cry,Crying_and_sobbing, Baby_cry_and_infant_cry,Screaming, Baby_cry_and_infant_cry,Screech, Baby_cry_and_infant_cry,Subway_and_metro_and_underground, Baby_laughter, Baby_laughter,Giggle, Baby_laughter,Laughter, Bark, Bark,Accelerating_and_revving_and_vroom, Bark,Ambulance_(siren), Bark,Bird_vocalization_and_bird_call_and_bird_song, Bark,Bleat,Sheep, Bark,Chatter,Sheep,Chirp_and_tweet, Bark,Chirp_and_tweet, Bark,Cricket, Bark,Cricket,Buzz,Bee_and_wasp_and_etc., Bark,Crow,Growling, Bark,Dog, Bark,Dog,Growling, Bark,Dog,Yip, Bark,Fill_(with_liquid),Trickle_and_dribble, Bark,Growling, Bark,Howl, Bark,Howl,Dog, Bark,Howl,Yip, Bark,Rain,Dog, Bark,Walk_and_footsteps, Bark,Walk_and_footsteps,Chirp_and_tweet, Bass_drum, Bass_drum,Didgeridoo, Bass_drum,Thump_and_thud, Bass_guitar ... (+180 more)
- First 10 rows/lines:
```text
{'fname': '64760', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '16399', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '16401', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '16402', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '16404', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '345111', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '64761', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '268259', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '64762', 'labels': 'Electric_guitar,Guitar', 'mids': '/m/02sgy,/m/0342h'}
{'fname': '160826', 'labels': 'Electric_guitar,Guitar', 'mids': '/m/02sgy,/m/0342h'}
```
Metadata `4060432/FSD50K.metadata/collection/collection_eval.csv`
- Columns/keys: ['fname', 'labels', 'mids']
- Total row/key count: 10231
- Unique label-like values in `fname`: 100, 100030, 100032, 1001, 100210, 100265, 100355, 100426, 1005, 100647, 1007, 100783, 100785, 100786, 100791, 1008, 100839, 100884, 100895, 100897, 1009, 100904, 1010, 1011, 101246, 1013, 101354, 101355, 1014, 101468, 101471, 101480, 101489, 101509, 1017, 101732, 101733, 101734, 101744, 101746, 101762, 101765, 101793, 101794, 101796, 101797, 101807, 101809, 101811, 101823, 101827, 101843, 101846, 101851, 101881, 101933, 101934, 101948, 102308, 102422, 102437, 102542, 102547, 102548, 102562, 102563, 102568, 102684, 102715, 102780, 102789, 1028, 102908, 102952, 102953, 102964, 102965, 102968, 102993, 102997, 103028, 103029, 103031, 103033, 103073, 103076, 1032, 103333, 103355, 103373, 103576, 103650, 1037, 104110, 1042, 104209, 104210, 104212, 104214, 104218, 104222, 104224, 104225, 104228, 104229, 104231, 104235, 104243, 104244, 104248, 104251, 104275, 104276, 104277, 104293, 1043, 104305, 1044, 104420, 104425 ... (+180 more)
- Unique label-like values in `labels`: Accordion, Accordion,Clarinet, Accordion,Conversation, Accordion,Glockenspiel, Accordion,Tambourine,Choir, Accordion,Violin_and_fiddle,Tambourine,Choir, Acoustic_guitar, Acoustic_guitar,Accordion, Acoustic_guitar,Bass_drum,Tambourine, Acoustic_guitar,Maraca, Acoustic_guitar,Strum, Acoustic_guitar,Synthesizer, Acoustic_guitar,Tambourine, Acoustic_guitar,Yell,Strum, Air_conditioning, Air_horn_and_truck_horn, Air_horn_and_truck_horn,Chatter,Ship, Air_horn_and_truck_horn,Cricket, Air_horn_and_truck_horn,Ocean,Ship, Air_horn_and_truck_horn,Ship, Air_horn_and_truck_horn,Wind_noise_(microphone),Ambulance_(siren),Police_car_(siren),Fire_engine_and_fire_truck_(siren), Aircraft, Aircraft,Chink_and_clink,Crackle, Aircraft,Waves_and_surf, Alarm, Alarm,Bell, Alarm,Bell,Chatter,Subway_and_metro_and_underground, Alarm,Bell,Engine,Children_shouting,Train,Chatter, Alarm,Bird,Skateboard,Child_speech_and_kid_speaking, Alarm,Bird_flight_and_flapping_wings,Chirp_and_tweet, Alarm,Chatter, Alarm,Chatter,Sliding_door,Slam, Alarm,Chatter,Subway_and_metro_and_underground, Alarm,Chatter,Subway_and_metro_and_underground,Sliding_door, Alarm,Door,Crowd,Bus, Alarm,Engine,Aircraft, Alarm,Engine,Bus, Alarm,Engine,Chatter,Fixed-wing_aircraft_and_airplane, Alarm,Engine,Fixed-wing_aircraft_and_airplane, Alarm,Engine,Mechanisms,Tools, Alarm,Engine,Shout,Fixed-wing_aircraft_and_airplane, Alarm,Engine,Train,Music, Alarm,Engine,Walk_and_footsteps,Mechanisms,Wind_noise_(microphone), Alarm,Female_speech_and_woman_speaking, Alarm,Female_speech_and_woman_speaking,Chatter,Subway_and_metro_and_underground,Sliding_door,Slam,Hiss, Alarm,Female_speech_and_woman_speaking,Sneeze,Chatter,Subway_and_metro_and_underground,Conversation, Alarm,Female_speech_and_woman_speaking,Speech_synthesizer,Subway_and_metro_and_underground, Alarm,Female_speech_and_woman_speaking,Subway_and_metro_and_underground, Alarm,Female_speech_and_woman_speaking,Subway_and_metro_and_underground,Sliding_door,Slam, Alarm,Female_speech_and_woman_speaking,Train, Alarm,Female_speech_and_woman_speaking,Train,Chatter,Sliding_door, Alarm,Fire,Tick-tock,Boom, Alarm,Frying_(food),Dishes_and_pots_and_pans, Alarm,Glockenspiel,Harp,Synthesizer, Alarm,Keys_jangling, Alarm,Marimba_and_xylophone, Alarm,Mechanisms,Chatter, Alarm,Microwave_oven, Alarm,Rattle,Thump_and_thud, Alarm,Screech,Sliding_door, Alarm,Sliding_door, Alarm,Sliding_door,Slam, Alarm,Snap, Alarm,Subway_and_metro_and_underground, Alarm,Subway_and_metro_and_underground,Sliding_door,Slam, Alarm,Synthesizer, Alarm,Thunder, Alarm,Train, Alarm,Train,Chatter,Sliding_door,Slam,Bicycle, Alarm,Train,Chink_and_clink, Alarm,Train,Sliding_door, Alarm,Train,Sliding_door,Hiss, Alarm,Train_horn, Alarm,Truck,Chatter,Motorcycle,Accelerating_and_revving_and_vroom, Alarm,Truck,Idling,Hiss, Alarm,Typewriter, Alarm,Walk_and_footsteps, Alarm,Wind_noise_(microphone), Alarm,Yell, Alarm,Zipper_(clothing), Alarm_clock, Alarm_clock,Buzz, Alarm_clock,Clicking, Alarm_clock,Tick-tock, Alto_saxophone, Ambulance_(siren), Ambulance_(siren),Car,Keys_jangling,Police_car_(siren),Traffic_noise_and_roadway_noise, Ambulance_(siren),Car_alarm,Police_car_(siren), Ambulance_(siren),Police_car_(siren), Ambulance_(siren),Traffic_noise_and_roadway_noise, Applause, Applause,Chatter,Cheering,Clapping, Applause,Chatter,Clapping, Applause,Chatter,Crowd,Cheering,Clapping,Whistling, Applause,Chatter,Crowd,Human_voice,Clapping, Applause,Chatter,Crowd,Laughter,Clapping, Applause,Chatter,Crowd,Laughter,Clapping,Child_speech_and_kid_speaking, Applause,Chatter,Crowd,Shout,Clapping, Applause,Cheering, Applause,Cheering,Clapping, Applause,Child_singing,Organ, Applause,Clapping, Applause,Clapping,Cough, Applause,Clapping,Whistling, Applause,Crowd, Applause,Crowd,Cheering,Clapping, Applause,Crowd,Cheering,Clapping,Whistling, Applause,Crowd,Cheering,Whistling, Applause,Crowd,Clapping, Applause,Crowd,Laughter, Applause,Crowd,Laughter,Clapping, Applause,Crowd,Whistling, Applause,Dishes_and_pots_and_pans,Whoop,Crowd,Cheering,Clapping, Applause,Knock,Laughter,Cheering,Clapping, Applause,Laughter,Cheering,Clapping, Applause,Screaming,Chatter,Crowd,Cheering,Clapping, Applause,Screaming,Crowd,Cheering,Yell,Clapping,Booing, Applause,Tap,Clapping, Applause,Thunder,Chatter,Cheering, Applause,Whoop,Ambulance_(siren),Clapping,Police_car_(siren) ... (+180 more)
- First 10 rows/lines:
```text
{'fname': '37199', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '175151', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '253463', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '329838', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '1277', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '30149', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '331398', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '333246', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '232924', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
{'fname': '42378', 'labels': 'Electric_guitar', 'mids': '/m/02sgy'}
```
Metadata `4060432/FSD50K.metadata/collection/vocabulary_collection_dev.csv`
- Columns/keys: ['0', 'Accelerating_and_revving_and_vroom', '/m/07q2z82']
- Total row/key count: 353
- First 10 rows/lines:
```text
{'0': '1', 'Accelerating_and_revving_and_vroom': 'Accordion', '/m/07q2z82': '/m/0mkg'}
{'0': '2', 'Accelerating_and_revving_and_vroom': 'Acoustic_guitar', '/m/07q2z82': '/m/042v_gx'}
{'0': '3', 'Accelerating_and_revving_and_vroom': 'Air_conditioning', '/m/07q2z82': '/m/025wky1'}
{'0': '4', 'Accelerating_and_revving_and_vroom': 'Air_horn_and_truck_horn', '/m/07q2z82': '/m/05x_td'}
{'0': '5', 'Accelerating_and_revving_and_vroom': 'Aircraft', '/m/07q2z82': '/m/0k5j'}
{'0': '6', 'Accelerating_and_revving_and_vroom': 'Aircraft_engine', '/m/07q2z82': '/m/014yck'}
{'0': '7', 'Accelerating_and_revving_and_vroom': 'Alarm', '/m/07q2z82': '/m/07pp_mv'}
{'0': '8', 'Accelerating_and_revving_and_vroom': 'Alarm_clock', '/m/07q2z82': '/m/046dlr'}
{'0': '9', 'Accelerating_and_revving_and_vroom': 'Alto_saxophone', '/m/07q2z82': '/m/02pprs'}
{'0': '10', 'Accelerating_and_revving_and_vroom': 'Ambulance_(siren)', '/m/07q2z82': '/m/012n7d'}
```
Metadata `4060432/FSD50K.metadata/collection/vocabulary_collection_eval.csv`
- Columns/keys: ['0', 'Accelerating_and_revving_and_vroom', '/m/07q2z82']
- Total row/key count: 361
- First 10 rows/lines:
```text
{'0': '1', 'Accelerating_and_revving_and_vroom': 'Accordion', '/m/07q2z82': '/m/0mkg'}
{'0': '2', 'Accelerating_and_revving_and_vroom': 'Acoustic_guitar', '/m/07q2z82': '/m/042v_gx'}
{'0': '3', 'Accelerating_and_revving_and_vroom': 'Air_conditioning', '/m/07q2z82': '/m/025wky1'}
{'0': '4', 'Accelerating_and_revving_and_vroom': 'Air_horn_and_truck_horn', '/m/07q2z82': '/m/05x_td'}
{'0': '5', 'Accelerating_and_revving_and_vroom': 'Aircraft', '/m/07q2z82': '/m/0k5j'}
{'0': '6', 'Accelerating_and_revving_and_vroom': 'Alarm', '/m/07q2z82': '/m/07pp_mv'}
{'0': '7', 'Accelerating_and_revving_and_vroom': 'Alarm_clock', '/m/07q2z82': '/m/046dlr'}
{'0': '8', 'Accelerating_and_revving_and_vroom': 'Alto_saxophone', '/m/07q2z82': '/m/02pprs'}
{'0': '9', 'Accelerating_and_revving_and_vroom': 'Ambulance_(siren)', '/m/07q2z82': '/m/012n7d'}
{'0': '10', 'Accelerating_and_revving_and_vroom': 'Applause', '/m/07q2z82': '/m/028ght'}
```
Metadata `4060432/FSD50K.metadata/dev_clips_info_FSD50K.json`
- Columns/keys: ['425873', '420945', '431448', '410343', '420940', '46138', '415439', '415438', '415431', '415430', '415433', '415432', '415435', '415434', '415436', '386305', '433886', '433887', '433880', '413106', '413105', '414454', '428338', '420102', '420105', '420107', '420108', '428484', '389695', '389690', '431628', '431627', '413485', '413480', '136859', '409742', '409743', '409740', '408296', '408294', '402238', '405020', '405029', '397651', '397657', '416680', '416686', '419503', '390864', '404087', '400504', '411087', '411082', '398738', '398739', '398734', '398736', '398737', '388478', '388479', '388477', '405842', '424317', '414040', '59201', '426204', '426205', '426206', '426207', '426208', '426209', '387714', '413738', '431299', '423648', '183599', '386498', '386496', '386492', '86265', '406255', '406257', '406256', '406251', '406250', '405787', '403555', '397449', '397448', '402204', '402203', '402209', '410656', '410657', '396316', '410655', '410658', '410659', '388849', '426040']
- Total row/key count: 40966
- First 10 rows/lines:
```text
{'425873': "{'title': 'water baltic sea foley water run in fast splashes heavy mono.wav', 'description': 'This sound effect was recorded with high quality sound equipment and comes from a sound library called Baltic Sea which is pack of 56 high quality audio files with a total length of 153 minutes. In this library you\\'ll find various ambients recorded on the shores of the Baltic Sea. You can use this sound for free, and if you need a whole package of ambients you can support my work and get it for 40$ her"}
{'420945': '{\'title\': \'Guitar Riff Pentablues.wav\', \'description\': \'A pentacblues guitar riff made with shreddage and FLStudio\\n\\nI am a musician too, please, check out my page at bandcamp:\\n\\n<a href="https://mrthenoronha.bandcamp.com/album/mythical-miles" rel="nofollow">https://mrthenoronha.bandcamp.com/album/mythical-miles</a>\', \'tags\': [\'strings\', \'rock\', \'shred\', \'metal\', \'overdriven\', \'rool\', \'king\', \'overdrive\', \'riffs\', \'guitar\', \'song\', \'electric\', \'slash\', \'music\', \'riff\', \'tube\', \'death\', \'string'}
{'431448': "{'title': 'punch_plastic_out_2.wav', 'description': 'Punch being pulled out of 64oz plastic container.', 'tags': ['tools', 'clunk', 'thunk', 'plastic', 'pop', 'punch'], 'license': 'http://creativecommons.org/publicdomain/zero/1.0/', 'uploader': 'StarTowerStudio'}"}
{'410343': "{'title': 'Roar.wav', 'description': 'This is a variation on a guttural growl I made to depict some giant bears in power armour. I recorded myself with a Zoom H1 and then fiddled with pitch shifts etc to get to this big cat noise (well, that was what I was aiming for)', 'tags': ['roaring', 'wild', 'animal', 'snarl', 'lion', 'tiger', 'growl', 'growling'], 'license': 'http://creativecommons.org/publicdomain/zero/1.0/', 'uploader': 'waxsocks'}"}
{'420940': "{'title': 'Washing machine 1', 'description': 'Recording of my washing machine.', 'tags': ['clothes', 'electrical', 'dryer', 'wash', 'machine', 'appliance', 'household', 'tumble', 'washing'], 'license': 'http://creativecommons.org/licenses/by/3.0/', 'uploader': 'Cigaro30'}"}
{'46138': '{\'title\': \'scooter.wav\', \'description\': \'scooter in a streetrecorded with MD sony and stereo microphone by myself to release sound landscapes for a theater play "Montedidio" (Théâtre de l\\\'Echappée, Laval France)\', \'tags\': [\'motor\', \'scooter\'], \'license\': \'http://creativecommons.org/licenses/by-nc/3.0/\', \'uploader\': \'arnaud coutancier\'}'}
{'415439': "{'title': 'Harp Run with Wind Chimes and Birds.wav', 'description': 'Melodic Snippets from recordings of me playing the Swar SanGam. This wonderful instrument is a combination of the Swarmandal and the Tampura. 15 harp strings and 4 Drone/Bass strings. \\r\\nIn these recordings I am only using the Swarmandal (Harp) part.\\r\\nIt is tuned to C sharp, but I have dropped the fourth note (F sharp) out of the scale.\\r\\n\\r\\nThere are four packs with lots of recordings in them, strums, plucks, short improv"}
{'415438': '{\'title\': \'Harp Up Run 1.wav\', \'description\': \'Melodic Snippets from recordings of me playing the Swar SanGam. This wonderful instrument is a combination of the Swarmandal and the Tampura. 15 harp strings and 4 Drone/Bass strings. \\r\\nIn these recordings I am only using the Swarmandal (Harp) part.\\r\\nIt is tuned to C sharp, but I have dropped the fourth note (F sharp) out of the scale.\\r\\n\\r\\nThere are four packs with lots of recordings in them, strums, plucks, short improvisations. \\r\\n"Short m'}
{'415431': '{\'title\': \'Harp Tinkly Riff 3.wav\', \'description\': \'Melodic Snippets from recordings of me playing the Swar SanGam. This wonderful instrument is a combination of the Swarmandal and the Tampura. 15 harp strings and 4 Drone/Bass strings. \\r\\nIn these recordings I am only using the Swarmandal (Harp) part.\\r\\nIt is tuned to C sharp, but I have dropped the fourth note (F sharp) out of the scale.\\r\\n\\r\\nThere are four packs with lots of recordings in them, strums, plucks, short improvisations. \\r\\n"Sh'}
{'415430': "{'title': 'Harp Strumming Riff 9.wav', 'description': 'Melodic Snippets from recordings of me playing the Swar SanGam. This wonderful instrument is a combination of the Swarmandal and the Tampura. 15 harp strings and 4 Drone/Bass strings. \\r\\nIn these recordings I am only using the Swarmandal (Harp) part.\\r\\nIt is tuned to C sharp, but I have dropped the fourth note (F sharp) out of the scale.\\r\\n\\r\\nThere are four packs with lots of recordings in them, strums, plucks, short improvisations. \\r\\n"}
```
Metadata `4060432/FSD50K.metadata/eval_clips_info_FSD50K.json`
- Columns/keys: ['391277', '392115', '411438', '395238', '425681', '431621', '369908', '426769', '426761', '411737', '390241', '390242', '390244', '388470', '425443', '425442', '425441', '425440', '425447', '425444', '392839', '392838', '414047', '391870', '125169', '395239', '417608', '408310', '149818', '433598', '433599', '433591', '433596', '433597', '400093', '400097', '204920', '407668', '238209', '404126', '396311', '414362', '425684', '425680', '397409', '391327', '403375', '266184', '394840', '415343', '390775', '413189', '412751', '412750', '407504', '389771', '402732', '389615', '389617', '389613', '389612', '238204', '408475', '406112', '404010', '90219', '429149', '389638', '396168', '408800', '408807', '408804', '418458', '412016', '415296', '394478', '406093', '406090', '380398', '412507', '412500', '394730', '394731', '424619', '424613', '424616', '431472', '408390', '408391', '409246', '385868', '31777', '406110', '413850', '159643', '424736', '412820', '431396', '431397', '431533']
- Total row/key count: 10231
- First 10 rows/lines:
```text
{'391277': "{'title': 'Spring Birds Forest 04 Amp.wav', 'description': 'An other birds singings recorded on the morning (Amplified)', 'tags': ['birdsong', 'bird', 'forest', 'environment', 'morning', 'nature', 'birds'], 'license': 'http://creativecommons.org/publicdomain/zero/1.0/', 'uploader': 'ANARKYA'}"}
{'392115': '{\'title\': \'Snap of fingers\', \'description\': "a snap of one\'s fingers", \'tags\': [\'fingers\', \'finger\', \'5maudio17\', \'uam\', \'fingersnap\'], \'license\': \'http://creativecommons.org/publicdomain/zero/1.0/\', \'uploader\': \'edton\'}'}
{'411438': "{'title': 'Pouring Water', 'description': 'A sound of Hot water pouring into a cup', 'tags': ['fill', 'can', 'beverage', 'glass', 'water', 'pour', 'drink', 'tea', 'cup', 'liquid', 'pouring'], 'license': 'http://creativecommons.org/publicdomain/zero/1.0/', 'uploader': 'edsward'}"}
{'395238': "{'title': 'Tearing papers.wav', 'description': 'Tearing papers with reverb.\\r\\n\\r\\nI used a Samson condenser microphone with Presonus AudioBox 22VSL Pre-amp. Software is Garageband, the reverb effect is from garageband. \\r\\nI took some old paper and started tearing it up. \\r\\nI needed this sound for my guided meditation where people tear up their old beliefs and get rid of old ideas and old feelings I use this sound effect to enhance the visualisation in my guided meditation. \\r\\n\\r\\nIt was reco"}
{'425681': "{'title': 'sigh.mp3', 'description': 'i recorded me sighing.', 'tags': ['sad', 'happy', 'sigh'], 'license': 'http://creativecommons.org/publicdomain/zero/1.0/', 'uploader': 'Camo1018'}"}
{'431621': "{'title': 'fart,bum,trumpet,poop.wav', 'description': 'A gluten intolerance has its advantages. \\nRecorded with a sennheiser 8060 into a mixpre3', 'tags': ['bum', 'farts', 'trumpet', 'poop', 'fart'], 'license': 'http://creativecommons.org/publicdomain/zero/1.0/', 'uploader': 'sorce'}"}
{'369908': '{\'title\': \'male speech\', \'description\': "Male speech with some specific pitch saying : \'audio signal processing for music applications\'", \'tags\': [\'man\', \'voice\', \'human\', \'entonation\', \'speech\', \'spoken\', \'male\', \'f-note\'], \'license\': \'http://creativecommons.org/publicdomain/zero/1.0/\', \'uploader\': \'germanio\'}'}
{'426769': "{'title': 'Drawer OPEN.wav', 'description': 'Drawer opening.', 'tags': ['bedroom', 'open', 'drawer'], 'license': 'http://creativecommons.org/publicdomain/zero/1.0/', 'uploader': 'cMilan'}"}
{'426761': "{'title': 'Académica', 'description': 'Supporters of the portuguese soccer team Associação Académica de Coimbra, while hosting Porto B team in a Honor Division game in april 2018.', 'tags': ['match', 'mica', 'supporter', 'soccer', 'team', 'coimbra', 'competition', 'acad', 'fans', 'cheers', 'briosa', 'game', 'football', 'portugal', 'sport'], 'license': 'http://creativecommons.org/publicdomain/zero/1.0/', 'uploader': 'cgoulao'}"}
{'411737': '{\'title\': \'keyboard.wav\', \'description\': "typing on my grandma\'s keyboard used dr-05 in her bedroom", \'tags\': [\'keystroke\', \'typing\', \'key\', \'computer\', \'keyboard\', \'type\', \'keys\', \'office\'], \'license\': \'http://creativecommons.org/publicdomain/zero/1.0/\', \'uploader\': \'jimmyfisher\'}'}
```
Metadata `4060432/FSD50K.metadata/pp_pnp_ratings_FSD50K.json`
- Columns/keys: ['338674', '393237', '338677', '262203', '262206', '63', '262213', '262214', '262233', '262237', '262240', '131171', '99', '131173', '100', '393323', '393326', '393333', '262265', '393342', '131199', '131200', '131201', '131202', '262275', '131203', '393349', '131206', '131205', '262278', '137', '262282', '262277', '393356', '136', '393358', '393359', '416854', '131218', '131226', '393374', '393376', '393379', '393380', '262308', '131237', '131236', '262312', '262313', '262314', '131239', '262322', '262326', '131258', '393402', '131264', '131265', '131266', '131267', '209', '393425', '221', '393442', '393444', '131303', '131305', '236', '237', '131309', '393454', '393456', '393455', '393458', '393459', '393460', '393461', '393457', '247', '393463', '393465', '393462', '131323', '393469', '393470', '262399', '262398', '393473', '262403', '131333', '263', '393479', '393490', '393491', '131348', '262421', '393492', '262423', '262424', '393497', '281']
- Total row/key count: 51197
- First 10 rows/lines:
```text
{'338674': "{'/m/0ghcn6': [1.0, 1.0], '/m/01z5f': [1.0, 1.0, -1]}"}
{'393237': "{'/m/07r660_': [1.0, 1.0, 0]}"}
{'338677': "{'/m/07pggtn': [0.5, 0.5]}"}
{'262203': "{'/m/012f08': [1.0, 1.0, -1], '/m/0k4j': [1.0, 1.0], '/t/dd00130': [1.0, 1.0]}"}
{'262206': "{'/m/07rrlb6': [1.0, 1.0]}"}
{'63': "{'/m/09l8g': [1.0, 1.0, 0.5]}"}
{'262213': "{'/m/01d380': [1.0]}"}
{'262214': "{'/m/020bb7': [1.0, 1.0]}"}
{'262233': "{'/m/05zppz': [1.0, 1.0, 1.0]}"}
{'262237': "{'/m/02zsn': [1.0, 1.0, 1.0, 1.0]}"}
```
### Step 4 - Audio Quality Check

- `4060432/FSD50K.dev_audio/10000.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
Sample-rate counts from sampled files: {'44100': 250}; formats: {'wav': 40966}

### Step 5 - Nature Subclass Mapping

- rain: FSD50K.ground_truth/dev.csv labels token in ['Rain']; clips=500; estimated 5s clips=1221; sample_rates={'44100': 500}
- sea_waves: NOT PRESENT
- wind: FSD50K.ground_truth/dev.csv labels token in ['Wind']; clips=294; estimated 5s clips=410; sample_rates={'44100': 294}
- crackling_fire: FSD50K.ground_truth/dev.csv labels token in ['Fire']; clips=385; estimated 5s clips=557; sample_rates={'44100': 385}

### FSD50K Requested Label Checks

- Nature raw label contains counts: {'Rain': 837, 'Water': 1711, 'Wind': 2905, 'Fire': 1124, 'Crackling': 0}
- Urban raw label contains counts: {'Car_horn': 115, 'Engine': 854, 'Siren': 77, 'Jackhammer': 0}

### Step 7 - Problem Flags

- clipping observed in sampled audio
Decision: USE WITH CAUTION


## ESC-50-master

### Step 1 - Identify Dataset

Folder structure up to 3 levels deep:

- `.circleci/`
- `.github/`
- `audio/`
- `meta/`
- `tests/`

Root non-audio files: ESC-50-master/.gitignore, ESC-50-master/LICENSE, ESC-50-master/README.md, ESC-50-master/esc50.gif, ESC-50-master/pytest.ini, ESC-50-master/requirements.txt

First 10 lines of `ESC-50-master/README.md`:
```text
## ESC-50: Dataset for Environmental Sound Classification

> ###### [Overview](#esc-50-dataset-for-environmental-sound-classification) | [Download](#download) | [Results](#results) | [Repository content](#repository-content) | [License](#license) | [Citing](#citing) | [Caveats](#caveats) | [Changelog](#changelog)
>
> <a href="https://circleci.com/gh/karoldvl/ESC-50"><img src="https://circleci.com/gh/karoldvl/ESC-50.svg?style=svg" /></a>&nbsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/license-CC%20BY--NC-blue.svg" />&nbsp;
<a href="https://github.com/karoldvl/ESC-50/archive/master.zip"><img src="https://img.shields.io/badge/download-.zip-ff69b4.svg" alt="Download" /></a>&nbsp;

<img src="esc50.gif" alt="ESC-50 clip preview" title="ESC-50 clip preview" align="right" />

```
First 10 lines of `ESC-50-master/requirements.txt`:
```text
librosa
matplotlib
numpy
pandas
seaborn
tqdm
```
Subfolder names: .circleci, .github, audio, meta, tests

### Step 2 - Count Audio Files

Total requested-extension audio files: 2000 (`wav/mp3/flac/ogg/m4a/aiff`). Including WebM/nonstandard: 2000.
Sample filenames:
- `ESC-50-master/audio/3-145719-A-17.wav` parent `ESC-50-master/audio`
- `ESC-50-master/audio/2-133889-A-30.wav` parent `ESC-50-master/audio`
- `ESC-50-master/audio/4-168155-A-15.wav` parent `ESC-50-master/audio`
- `ESC-50-master/audio/3-164595-A-15.wav` parent `ESC-50-master/audio`
- `ESC-50-master/audio/4-172366-A-37.wav` parent `ESC-50-master/audio`
Unique parent folder names containing requested audio: audio

### Step 3 - Inspect Metadata Files

Metadata `ESC-50-master/README.md`
- Columns/keys: []
- Total row/key count: 208
- First 10 rows/lines:
```text
## ESC-50: Dataset for Environmental Sound Classification

> ###### [Overview](#esc-50-dataset-for-environmental-sound-classification) | [Download](#download) | [Results](#results) | [Repository content](#repository-content) | [License](#license) | [Citing](#citing) | [Caveats](#caveats) | [Changelog](#changelog)
>
> <a href="https://circleci.com/gh/karoldvl/ESC-50"><img src="https://circleci.com/gh/karoldvl/ESC-50.svg?style=svg" /></a>&nbsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/license-CC%20BY--NC-blue.svg" />&nbsp;
<a href="https://github.com/karoldvl/ESC-50/archive/master.zip"><img src="https://img.shields.io/badge/download-.zip-ff69b4.svg" alt="Download" /></a>&nbsp;

<img src="esc50.gif" alt="ESC-50 clip preview" title="ESC-50 clip preview" align="right" />

```
Metadata `ESC-50-master/meta/esc50.csv`
- Columns/keys: ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
- Total row/key count: 2000
- Unique label-like values in `filename`: 1-100032-A-0.wav, 1-100038-A-14.wav, 1-100210-A-36.wav, 1-100210-B-36.wav, 1-101296-A-19.wav, 1-101296-B-19.wav, 1-101336-A-30.wav, 1-101404-A-34.wav, 1-103298-A-9.wav, 1-103995-A-30.wav, 1-103999-A-30.wav, 1-104089-A-22.wav, 1-104089-B-22.wav, 1-105224-A-22.wav, 1-110389-A-0.wav, 1-110537-A-22.wav, 1-115521-A-19.wav, 1-115545-A-48.wav, 1-115545-B-48.wav, 1-115545-C-48.wav, 1-115546-A-48.wav, 1-115920-A-22.wav, 1-115920-B-22.wav, 1-115921-A-22.wav, 1-116765-A-41.wav, 1-11687-A-47.wav, 1-118206-A-31.wav, 1-118559-A-17.wav, 1-119125-A-45.wav, 1-121951-A-8.wav, 1-12653-A-15.wav, 1-12654-A-15.wav, 1-12654-B-15.wav, 1-13571-A-46.wav, 1-13572-A-46.wav, 1-13613-A-37.wav, 1-137-A-32.wav, 1-137296-A-16.wav, 1-14262-A-37.wav, 1-155858-A-25.wav, 1-155858-B-25.wav, 1-155858-C-25.wav, 1-155858-D-25.wav, 1-155858-E-25.wav, 1-155858-F-25.wav, 1-15689-A-4.wav, 1-15689-B-4.wav, 1-160563-A-48.wav, 1-160563-B-48.wav, 1-16568-A-3.wav, 1-16746-A-15.wav, 1-17092-A-27.wav, 1-17092-B-27.wav, 1-17124-A-43.wav, 1-17150-A-12.wav, 1-172649-A-40.wav, 1-172649-B-40.wav, 1-172649-C-40.wav, 1-172649-D-40.wav, 1-172649-E-40.wav, 1-172649-F-40.wav, 1-17295-A-29.wav, 1-17367-A-10.wav, 1-17565-A-12.wav, 1-17585-A-7.wav, 1-17742-A-12.wav, 1-17808-A-12.wav, 1-17808-B-12.wav, 1-1791-A-26.wav, 1-17970-A-4.wav, 1-18074-A-6.wav, 1-18074-B-6.wav, 1-181071-A-40.wav, 1-181071-B-40.wav, 1-18527-A-44.wav, 1-18527-B-44.wav, 1-18631-A-23.wav, 1-18655-A-31.wav, 1-187207-A-20.wav, 1-18755-A-4.wav, 1-18755-B-4.wav, 1-18757-A-4.wav, 1-18810-A-49.wav, 1-19026-A-43.wav, 1-19111-A-24.wav, 1-19118-A-24.wav, 1-19501-A-7.wav, 1-196660-A-8.wav, 1-196660-B-8.wav, 1-19840-A-36.wav, 1-19872-A-36.wav, 1-19872-B-36.wav, 1-19898-A-41.wav, 1-19898-B-41.wav, 1-19898-C-41.wav, 1-20133-A-39.wav, 1-202111-A-3.wav, 1-20545-A-28.wav, 1-20736-A-18.wav, 1-208757-A-2.wav, 1-208757-B-2.wav, 1-208757-C-2.wav, 1-208757-D-2.wav, 1-208757-E-2.wav, 1-211527-A-20.wav, 1-211527-B-20.wav, 1-211527-C-20.wav, 1-21189-A-10.wav, 1-21421-A-46.wav, 1-21896-A-35.wav, 1-21934-A-38.wav, 1-21935-A-38.wav, 1-223162-A-25.wav, 1-22694-A-20.wav, 1-22694-B-20.wav, 1-22804-A-46.wav, 1-22882-A-44.wav, 1-23094-A-15.wav, 1-23094-B-15.wav, 1-23222-A-19.wav ... (+180 more)
- Unique label-like values in `target`: 0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 6, 7, 8, 9
- Unique label-like values in `category`: airplane, breathing, brushing_teeth, can_opening, car_horn, cat, chainsaw, chirping_birds, church_bells, clapping, clock_alarm, clock_tick, coughing, cow, crackling_fire, crickets, crow, crying_baby, dog, door_wood_creaks, door_wood_knock, drinking_sipping, engine, fireworks, footsteps, frog, glass_breaking, hand_saw, helicopter, hen, insects, keyboard_typing, laughing, mouse_click, pig, pouring_water, rain, rooster, sea_waves, sheep, siren, sneezing, snoring, thunderstorm, toilet_flush, train, vacuum_cleaner, washing_machine, water_drops, wind
- First 10 rows/lines:
```text
{'filename': '1-100032-A-0.wav', 'fold': '1', 'target': '0', 'category': 'dog', 'esc10': 'True', 'src_file': '100032', 'take': 'A'}
{'filename': '1-100038-A-14.wav', 'fold': '1', 'target': '14', 'category': 'chirping_birds', 'esc10': 'False', 'src_file': '100038', 'take': 'A'}
{'filename': '1-100210-A-36.wav', 'fold': '1', 'target': '36', 'category': 'vacuum_cleaner', 'esc10': 'False', 'src_file': '100210', 'take': 'A'}
{'filename': '1-100210-B-36.wav', 'fold': '1', 'target': '36', 'category': 'vacuum_cleaner', 'esc10': 'False', 'src_file': '100210', 'take': 'B'}
{'filename': '1-101296-A-19.wav', 'fold': '1', 'target': '19', 'category': 'thunderstorm', 'esc10': 'False', 'src_file': '101296', 'take': 'A'}
{'filename': '1-101296-B-19.wav', 'fold': '1', 'target': '19', 'category': 'thunderstorm', 'esc10': 'False', 'src_file': '101296', 'take': 'B'}
{'filename': '1-101336-A-30.wav', 'fold': '1', 'target': '30', 'category': 'door_wood_knock', 'esc10': 'False', 'src_file': '101336', 'take': 'A'}
{'filename': '1-101404-A-34.wav', 'fold': '1', 'target': '34', 'category': 'can_opening', 'esc10': 'False', 'src_file': '101404', 'take': 'A'}
{'filename': '1-103298-A-9.wav', 'fold': '1', 'target': '9', 'category': 'crow', 'esc10': 'False', 'src_file': '103298', 'take': 'A'}
{'filename': '1-103995-A-30.wav', 'fold': '1', 'target': '30', 'category': 'door_wood_knock', 'esc10': 'False', 'src_file': '103995', 'take': 'A'}
```
Metadata `ESC-50-master/requirements.txt`
- Columns/keys: []
- Total row/key count: 6
- First 10 rows/lines:
```text
librosa
matplotlib
numpy
pandas
seaborn
tqdm
```
### Step 4 - Audio Quality Check

- `ESC-50-master/audio/1-100032-A-0.wav`: format=wav, ok=False, sample_rate=None, duration=None, channels=None, silent=None, clipped=None, error=cannot cache function '__o_fold': no locator available for file '/home/abhi-ubuntu-pc/anaconda3/lib/python3.13/site-packages/librosa/core/notation.py'
Sample-rate counts from sampled files: {'44100': 250}; formats: {'wav': 2000}

### Step 5 - Nature Subclass Mapping

- rain: meta/esc50.csv category in ['rain']; clips=40; estimated 5s clips=40; sample_rates={'44100': 40}
- sea_waves: meta/esc50.csv category in ['sea_waves']; clips=40; estimated 5s clips=40; sample_rates={'44100': 40}
- wind: meta/esc50.csv category in ['wind']; clips=40; estimated 5s clips=40; sample_rates={'44100': 40}
- crackling_fire: meta/esc50.csv category in ['crackling_fire']; clips=40; estimated 5s clips=40; sample_rates={'44100': 40}

### Step 7 - Problem Flags

- clipping observed in sampled audio
Decision: USE WITH CAUTION


## Step 8 - Summary Table

| Folder | Dataset Name | Total Audio Files | Nature Subclasses Found | Urban Subclasses Found | Label Mechanism | Sample Rate | Decision: Use/Skip |
|---|---:|---:|---|---|---|---|---|
| `rain_dataset` | rain_dataset | 0 | None | None | metadata | unknown | SKIP |
| `audio_noise_dataset` | audio_noise_dataset | 0 | None | None | none | unknown | SKIP |
| `FSC22_forest` | FSC22_forest | 2025 | rain:75/75x5s, wind:75/75x5s, crackling_fire:75/75x5s | None | metadata | 44100:121, 48000:85, 24000:29, 16000:2, 96000:13 | USE WITH CAUTION |
| `forest_wild_fire_sound_dataset` | forest_wild_fire_sound_dataset | 289 | crackling_fire:289/2878x5s | None | folder/path | 44100:250 | USE WITH CAUTION |
| `freefield1010` | freefield1010 | 7690 | rain:434/868x5s, sea_waves:400/800x5s, wind:428/856x5s, crackling_fire:70/140x5s | None | metadata | 44100:250 | USE |
| `urbansound8k` | urbansound8k | 8732 | None | car_horn:429/0x5s, engine_idling:1000/0x5s, siren:929/0x5s, jackhammer:1000/0x5s | metadata | 44100:147, 48000:24, 11025:2, 24000:1, 16000:3, 96000:2, 22050:3 | SKIP |
| `99Sounds Nature Sounds` | 99Sounds Nature Sounds | 83 | rain:6/108x5s, wind:14/174x5s, sea_waves:4/63x5s | None | metadata | 96000:8, 192000:17 | USE |
| `4060432` | 4060432 | 40966 | rain:500/1221x5s, wind:294/410x5s, crackling_fire:385/557x5s | car_horn:115/110x5s, engine_idling:854/1566x5s, siren:77/182x5s | metadata | 44100:250 | USE WITH CAUTION |
| `ESC-50-master` | ESC-50-master | 2000 | rain:40/40x5s, sea_waves:40/40x5s, wind:40/40x5s, crackling_fire:40/40x5s | None | metadata | 44100:250 | USE WITH CAUTION |