# Dataset review

## GuitarSet 1.1.0

GuitarSet contains 360 roughly 30-second acoustic-guitar excerpts with time-aligned chords, notes, beats, downbeats, playing style, and string/fret information. The official Zenodo record includes annotation, mono microphone audio, mono pickup mix, original hexaphonic audio, and debleeded hexaphonic audio. Thus it is not annotation-only. Use version 1.1.0 and review its known timing/annotation errors. The paper states CC BY 4.0; preserve citation and verify the current record rights before deployment.

For leakage control, all channels and renderings of one player/performance must share one `group_id`. It is useful for guitar-only evaluation and auxiliary training, but by itself does not represent vocal/drum/bass mixtures.

## Mixed popular music

BTC reports Isophonics, Robbie Williams, and UsPop2002. The BTC README explicitly says these datasets consist of chord label files and do not include the copyrighted audio used in the work. The USPop/RWC chord-annotation repository likewise says it distributes annotations only. Users must lawfully source audio and verify recording/version alignment.

McGill Billboard 2.0 distributes CC0 chord/structure annotations and non-invertible features, not original copyrighted audio. It reports 890 annotated slots covering 740 distinct songs, so duplicate songs/slots must be grouped before splitting. Its beat-level chord timings include interpolation; this should be documented in evaluation.

Recommended protocol:

1. Resolve lawful audio and annotation version alignment.
2. Assign a stable recording/group ID before any chunking or augmentation.
3. Split original recordings into train/validation/test.
4. Put alternate channels, encodings, overlapping excerpts, and augmentations in the same group.
5. Compute normalization only from train features.
6. Use GuitarSet and mixed music as separate test strata; report N included/excluded and unsupported-label ratio.

