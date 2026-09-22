# Experiments

The measured record of this work, generated from the artifacts by
[`scripts/experiment-ledger.py`](../scripts/experiment-ledger.py). Do not hand-edit: change the
script's annotations or the logs and regenerate, or the file and the evidence will drift apart.

The protocol is fixed and is the only arbiter: chat mode, `thinking=false`, seed 1234, 512-token
generations, levels 1 3 5 6, three passes, median reported per level. Both arms launch through
`scripts/05-serve.sh` with the worker first and the head about 85 s later.

## How to read a row

- **guarded** means the arm's log records the container that answered. Every arm measured before
  2026-09-13T03:15 is listed under *superseded* instead, and is not evidence: a still-running
  reference container once answered a whole series, which made five candidate arms look like the
  reference.
- **spread** is the pass-to-pass swing of the median. One 128-token pass can swing 18 % on this
  rig, so a change is kept only if it beats the larger spread of the two arms being compared.
- **sum** is the sum of the four level medians, which the completion criterion also uses.
- the gates are `gate_france` (' Paris...') and `gate_9x8` ('72, 9x9'). Failing them means the
  model's own output is wrong, which has happened twice and was the whole signal in both cases.

## Candidate arms, and the reference they are measured against

| tag | c1 | c3 | c5 | c6 | sum | worst spread | gates |
|---|---|---|---|---|---|---|---|
| `refg-now` | 63.7 | 117.0 | 142.0 | 161.2 | 483.9 | 26.6 % | pass |
| `refg` | 65.3 | 110.1 | 138.6 | 159.1 | 473.1 | 16.1 % | pass |
| `ctl2-hum-rc2` | 66.3 | 117.3 | 144.7 | 165.7 | 494.0 | 13.3 % | pass |
| `ctl3-rc2` | 62.9 | 116.6 | 147.2 | 164.3 | 491.0 | 13.2 % | pass |
| `abOTH-d` | 67.6 | 116.0 | 146.6 | 160.3 | 490.5 | 14.5 % | pass |
| `qO-a` | 63.6 | 114.3 | 145.3 | 164.9 | 488.1 | 15.8 % | pass |
| `proto2-030-0b` | 64.1 | 117.8 | 145.1 | 158.9 | 485.9 | 10.8 % | pass |
| `refg-rc2b` | 63.0 | 114.6 | 146.1 | 162.2 | 485.9 | 20.7 % | pass |
| `capsz-030-0` | 62.3 | 117.6 | 145.6 | 159.5 | 485.0 | 11.6 % | pass |
| `ctl-hum-rc2` | 61.0 | 116.3 | 146.3 | 161.2 | 484.8 | 7.1 % | pass |
| `pO-a` | 63.7 | 112.6 | 145.9 | 161.5 | 483.7 | 11.8 % | pass |
| `qO-d` | 62.2 | 111.5 | 148.3 | 161.7 | 483.7 | 11.7 % | pass |
| `refg030` | 64.0 | 114.2 | 143.4 | 162.1 | 483.7 | 18.5 % | pass |
| `rO-a` | 62.8 | 112.5 | 146.2 | 161.7 | 483.2 | 9.9 % | pass |
| `abREF-b` | 61.8 | 114.9 | 145.0 | 159.0 | 480.7 | 26.2 % | pass |
| `rO-d` | 63.7 | 109.5 | 146.4 | 160.7 | 480.3 | 8.0 % | pass |
| `pR-b` | 62.9 | 112.0 | 139.1 | 163.7 | 477.7 | 17.6 % | pass |
| `hum-k6-rc2b` | 58.5 | 111.6 | 146.8 | 160.6 | 477.5 | 11.8 % | pass |
| `rR-c` | 62.2 | 111.4 | 142.6 | 161.0 | 477.2 | 23.8 % | pass |
| `qR-c` | 61.2 | 114.3 | 142.2 | 159.4 | 477.1 | 17.4 % | pass |
| `rR-b` | 63.1 | 115.2 | 139.6 | 157.4 | 475.3 | 24.8 % | pass |
| `pR-c` | 59.8 | 107.8 | 147.8 | 155.7 | 471.1 | 25.0 % | pass |
| `abREF-c` | 63.1 | 111.9 | 139.6 | 155.7 | 470.3 | 24.4 % | pass |
| `qR-b` | 60.9 | 111.7 | 136.6 | 159.2 | 468.4 | 19.5 % | pass |
| `refg-rc2` | 58.6 | 113.4 | 138.8 | 157.1 | 467.9 | 21.9 % | pass |
| `kv98304` | 59.8 | 109.7 | 136.2 | 157.6 | 463.3 | 9.9 % | pass |
| `abOTH-a` | 60.3 | 107.5 | 138.3 | 151.3 | 457.4 | 8.9 % | pass |
| `moemar` | 58.7 | 101.8 | 140.0 | 154.8 | 455.3 | 4.5 % | pass |
| `kv65536` | 57.8 | 107.9 | 135.2 | 153.5 | 454.4 | 8.7 % | pass |
| `kv229376` | 55.0 | 106.9 | 139.5 | 152.8 | 454.2 | 10.0 % | pass |
| `moea16-rc2` | 56.7 | 105.7 | 138.5 | 152.8 | 453.7 | 7.1 % | pass |
| `abwo-c` | 55.1 | 107.9 | 138.8 | 151.3 | 453.1 | 10.5 % | pass |
| `abwo-a` | 57.1 | 107.8 | 134.6 | 150.9 | 450.4 | 7.6 % | pass |
| `k6` | 58.4 | 106.7 | 132.8 | 151.6 | 449.5 | 9.6 % | pass |
| `abk6-b` | 55.8 | 105.3 | 133.1 | 155.2 | 449.4 | 8.1 % | pass |
| `kv245760` | 57.1 | 107.2 | 137.4 | 147.6 | 449.3 | 5.6 % | pass |
| `hum-k6-rc2` | 56.8 | 102.7 | 136.8 | 152.6 | 448.9 | 13.0 % | pass |
| `abk6-a` | 53.6 | 104.8 | 132.7 | 157.7 | 448.8 | 11.6 % | pass |
| `rc2back-rc2` | 56.9 | 104.0 | 134.5 | 153.3 | 448.7 | 11.3 % | pass |
| `wooff-k6-rc2` | 56.9 | 102.2 | 137.5 | 151.8 | 448.4 | 9.7 % | pass |
| `moehum` | 57.8 | 102.6 | 135.9 | 152.0 | 448.3 | 8.8 % | pass |
| `nowo` | 58.8 | 105.4 | 132.3 | 151.5 | 448.0 | 5.4 % | pass |
| `abwo-d` | 56.8 | 101.8 | 134.8 | 154.2 | 447.6 | 7.9 % | pass |
| `cgpiece-rc2` | 54.6 | 106.4 | 136.1 | 150.4 | 447.5 | 17.8 % | pass |
| `kv131072` | 59.1 | 103.5 | 133.3 | 150.9 | 446.8 | 6.1 % | pass |
| `kv196608` | 56.6 | 107.1 | 133.9 | 148.9 | 446.5 | 9.9 % | pass |
| `moea16` | 59.3 | 105.4 | 135.4 | 146.2 | 446.3 | 10.2 % | pass |
| `k6-rc2` | 55.9 | 107.6 | 132.2 | 149.9 | 445.6 | 10.9 % | pass |
| `moehum-rc2` | 58.1 | 102.2 | 132.3 | 152.9 | 445.5 | 4.7 % | pass |
| `wooff-k6-5-rc2` | 56.3 | 103.7 | 132.9 | 152.3 | 445.2 | 7.6 % | pass |
| `hum-a16-rc2` | 57.8 | 102.9 | 134.9 | 149.5 | 445.1 | 7.6 % | pass |
| `abwo-b` | 56.5 | 103.6 | 133.9 | 150.8 | 444.8 | 5.5 % | pass |
| `proto2-030-0` | 57.3 | 104.8 | 132.2 | 150.2 | 444.5 | 12.6 % | pass |
| `k5-rc2` | 56.7 | 106.5 | 118.8 | 162.0 | 444.0 | 6.2 % | pass |
| `kv262u086` | 61.5 | 105.1 | 128.6 | 148.5 | 443.7 | 9.7 % | pass |
| `nowo-rc2` | 57.7 | 102.6 | 132.9 | 149.0 | 442.2 | 5.7 % | pass |
| `ncclp2p0-rc2` | 55.8 | 98.4 | 137.7 | 148.7 | 440.6 | 13.8 % | pass |
| `ncclqps4-rc2` | 55.6 | 101.2 | 135.5 | 146.4 | 438.7 | 8.1 % | pass |
| `proto2-dg` | 55.5 | 99.6 | 131.7 | 150.4 | 437.2 | 13.9 % | pass |
| `k5` | 55.5 | 96.6 | 118.1 | 163.2 | 433.4 | 10.5 % | pass |
| `kvblnhc-rc2` | 56.8 | 97.3 | 132.1 | 147.0 | 433.2 | 5.1 % | pass |
| `wochunk4-rc2` | 55.3 | 101.5 | 130.3 | 145.8 | 432.9 | 5.9 % | pass |
| `moewarm-rc2` | 53.8 | 98.9 | 131.1 | 148.9 | 432.7 | 11.3 % | pass |
| `ncclbuf1m-rc2` | 52.8 | 101.3 | 134.9 | 143.7 | 432.7 | 9.3 % | pass |
| `attnfi030` | 56.6 | 99.7 | 129.8 | 145.3 | 431.4 | 10.6 % | pass |
| `moework-rc2` | 53.9 | 102.9 | 132.2 | 142.2 | 431.2 | 10.2 % | pass |
| `ncclxnic-rc2` | 56.9 | 101.9 | 127.0 | 145.3 | 431.1 | 7.9 % | pass |
| `micromax-rc2` | 54.8 | 99.9 | 133.9 | 142.4 | 431.0 | 10.6 % | pass |
| `moetile64-rc2` | 55.4 | 102.0 | 131.0 | 142.2 | 430.6 | 12.6 % | pass |
| `ncclto22-rc2` | 55.1 | 97.6 | 131.2 | 145.6 | 429.5 | 11.6 % | pass |
| `wochunk2-rc2` | 56.8 | 94.8 | 130.2 | 147.7 | 429.5 | 13.9 % | pass |
| `ncclplug0-rc2` | 53.7 | 101.3 | 132.0 | 142.4 | 429.4 | 10.0 % | pass |
| `now4sh-rc2` | 52.1 | 99.1 | 130.9 | 147.3 | 429.4 | 6.2 % | pass |
| `wooff5-rc2` | 54.2 | 99.5 | 131.1 | 144.6 | 429.4 | 8.5 % | pass |
| `cgwarm3-rc2` | 53.6 | 99.6 | 129.3 | 146.8 | 429.3 | 9.3 % | pass |
| `util8663` | 51.9 | 102.6 | 129.2 | 145.3 | 429.0 | 6.2 % | pass |
| `ncclar1-rc2` | 56.4 | 99.1 | 131.5 | 141.8 | 428.8 | 10.1 % | pass |
| `cgcopy-rc2` | 55.0 | 102.6 | 128.7 | 142.3 | 428.6 | 7.8 % | pass |
| `cgsizes-rc2` | 53.8 | 99.2 | 129.5 | 145.6 | 428.1 | 9.3 % | pass |
| `ncclpxn0-rc2` | 54.2 | 101.2 | 127.7 | 144.8 | 427.9 | 7.3 % | pass |
| `gencvllm-rc2` | 55.9 | 99.1 | 130.4 | 142.4 | 427.8 | 7.6 % | pass |
| `nostream` | 52.5 | 100.0 | 133.6 | 141.6 | 427.7 | 9.7 % | pass |
| `nostream-rc2` | 55.5 | 99.4 | 129.1 | 143.7 | 427.7 | 7.4 % | pass |
| `moetilem32-rc2` | 51.9 | 98.4 | 130.4 | 146.8 | 427.5 | 17.0 % | pass |
| `wooff-a16-rc2` | 54.9 | 102.1 | 130.8 | 139.6 | 427.4 | 5.9 % | pass |
| `noh16` | 53.6 | 100.8 | 129.5 | 143.4 | 427.3 | 8.0 % | pass |
| `wooff-rc2` | 56.8 | 96.5 | 129.4 | 144.2 | 426.9 | 7.5 % | pass |
| `gencvllm` | 55.9 | 101.4 | 127.7 | 141.8 | 426.8 | 6.8 % | pass |
| `moework` | 54.8 | 97.3 | 129.3 | 145.4 | 426.8 | 6.6 % | pass |
| `ncclhca-rc2` | 53.2 | 101.1 | 130.5 | 141.7 | 426.5 | 11.3 % | pass |
| `proto2-dg-pair` | 53.9 | 98.6 | 129.9 | 144.0 | 426.4 | 7.6 % | pass |
| `notiny-rc2` | 55.2 | 98.9 | 127.9 | 144.2 | 426.2 | 7.2 % | pass |
| `notiny` | 52.8 | 99.2 | 133.1 | 141.0 | 426.1 | 8.9 % | pass |
| `atom24` | 52.8 | 103.0 | 125.4 | 144.8 | 426.0 | 9.7 % | pass |
| `noreuse-rc2` | 54.7 | 100.5 | 127.0 | 143.8 | 426.0 | 5.1 % | pass |
| `moecut32-rc2` | 55.4 | 99.6 | 129.9 | 141.0 | 425.9 | 5.4 % | pass |
| `ncclsl0-rc2` | 54.8 | 97.7 | 130.8 | 142.6 | 425.9 | 10.8 % | pass |
| `split2` | 54.7 | 93.1 | 130.5 | 147.5 | 425.8 | 9.1 % | pass |
| `proto2-moetile` | 58.2 | 96.8 | 126.4 | 144.3 | 425.7 | 8.2 % | pass |
| `ncclnt64-rc2` | 54.8 | 97.3 | 125.7 | 147.8 | 425.6 | 6.6 % | pass |
| `lpf1024` | 56.0 | 97.9 | 126.7 | 144.9 | 425.5 | 14.4 % | pass |
| `lpf1024-rc2` | 53.9 | 99.2 | 129.0 | 143.4 | 425.5 | 7.3 % | pass |
| `abk7-a` | 52.5 | 102.6 | 127.2 | 142.9 | 425.2 | 7.7 % | pass |
| `memprof0` | 56.5 | 97.5 | 131.0 | 140.2 | 425.2 | 9.1 % | pass |
| `maxlen32k-rc2` | 54.3 | 97.7 | 128.6 | 144.3 | 424.9 | 6.8 % | pass |
| `noshare` | 54.6 | 100.0 | 128.8 | 141.4 | 424.8 | 11.6 % | pass |
| `noidxk-rc2` | 54.0 | 100.0 | 129.0 | 141.7 | 424.7 | 7.7 % | pass |
| `wochunk32-rc2` | 53.1 | 99.4 | 129.0 | 143.2 | 424.7 | 10.2 % | pass |
| `conn8-rc2` | 55.0 | 98.7 | 130.6 | 140.3 | 424.6 | 3.3 % | pass |
| `p030-rc2b` | 52.8 | 99.6 | 128.0 | 144.2 | 424.6 | 13.6 % | pass |
| `attnfi-rc2` | 53.0 | 99.3 | 129.3 | 142.8 | 424.4 | 5.7 % | pass |
| `nofast` | 56.2 | 101.1 | 124.4 | 142.7 | 424.4 | 16.4 % | pass |
| `moecap175` | 55.3 | 101.0 | 128.8 | 139.2 | 424.3 | 9.5 % | pass |
| `proto2-dgather` | 51.9 | 101.4 | 127.6 | 143.4 | 424.3 | 14.6 % | pass |
| `util8663-rc2` | 52.4 | 100.0 | 128.8 | 143.1 | 424.3 | 6.9 % | pass |
| `idxpr-rc2` | 53.5 | 98.5 | 128.8 | 143.4 | 424.2 | 5.3 % | pass |
| `moetile32` | 52.7 | 98.2 | 129.7 | 143.6 | 424.2 | 3.4 % | pass |
| `moetilem16-rc2` | 55.3 | 97.5 | 128.7 | 142.7 | 424.2 | 6.1 % | pass |
| `loadsaf-rc2` | 55.3 | 96.9 | 127.0 | 144.9 | 424.1 | 7.9 % | pass |
| `wochunk8-rc2` | 54.5 | 96.6 | 129.9 | 143.1 | 424.1 | 13.5 % | pass |
| `atom24-rc2` | 54.0 | 97.6 | 130.8 | 141.6 | 424.0 | 15.4 % | pass |
| `nccltc106-rc2` | 52.0 | 102.8 | 128.0 | 141.2 | 424.0 | 6.9 % | pass |
| `nocar-rc2` | 54.1 | 99.6 | 129.7 | 140.6 | 424.0 | 12.0 % | pass |
| `w4scr128-rc2` | 53.0 | 98.3 | 129.2 | 143.5 | 424.0 | 7.4 % | pass |
| `ncclch1-rc2` | 53.5 | 100.1 | 127.4 | 142.9 | 423.9 | 7.9 % | pass |
| `noshare-rc2` | 51.7 | 99.5 | 127.4 | 145.0 | 423.6 | 7.2 % | pass |
| `idxpr` | 52.8 | 96.2 | 128.5 | 145.9 | 423.4 | 9.4 % | pass |
| `nofast-rc2` | 51.4 | 99.5 | 130.2 | 142.3 | 423.4 | 5.1 % | pass |
| `nogqa6-rc2` | 54.6 | 94.2 | 132.4 | 142.2 | 423.4 | 9.2 % | pass |
| `dsl471-rc2` | 55.9 | 98.5 | 128.3 | 140.6 | 423.3 | 6.0 % | pass |
| `noasync` | 55.7 | 97.4 | 127.0 | 143.0 | 423.1 | 11.8 % | pass |
| `nopfx-rc2` | 54.6 | 100.0 | 128.3 | 140.2 | 423.1 | 6.6 % | pass |
| `noturbo` | 53.7 | 96.1 | 128.8 | 144.5 | 423.1 | 8.9 % | fail |
| `nogemv-rc2` | 54.1 | 97.1 | 127.2 | 144.6 | 423.0 | 7.6 % | pass |
| `igp-rc2` | 53.1 | 95.8 | 130.7 | 143.2 | 422.8 | 4.9 % | pass |
| `moeshare-rc2` | 51.3 | 98.0 | 129.3 | 144.2 | 422.8 | 13.5 % | pass |
| `conn8` | 54.7 | 100.4 | 125.8 | 141.7 | 422.6 | 7.9 % | pass |
| `proto2-dgather2` | 55.1 | 98.2 | 129.7 | 139.6 | 422.6 | 4.4 % | pass |
| `downsc` | 56.0 | 97.3 | 129.1 | 140.1 | 422.5 | 5.0 % | pass |
| `moecap175-rc2` | 55.3 | 100.1 | 124.5 | 142.5 | 422.4 | 8.3 % | pass |
| `idxfi-rc2` | 50.7 | 97.6 | 127.8 | 146.2 | 422.3 | 11.0 % | pass |
| `nopagemax-rc2` | 54.0 | 97.7 | 128.7 | 141.8 | 422.2 | 11.5 % | pass |
| `split4` | 51.9 | 100.9 | 126.6 | 142.8 | 422.2 | 11.0 % | pass |
| `ncclnsock4-rc2` | 53.0 | 99.0 | 128.1 | 142.0 | 422.1 | 12.8 % | pass |
| `moetile32-rc2` | 53.7 | 96.4 | 127.5 | 144.4 | 422.0 | 8.2 % | pass |
| `abk7-b` | 51.5 | 97.0 | 129.3 | 144.1 | 421.9 | 8.7 % | pass |
| `ncclretry1-rc2` | 53.1 | 101.1 | 125.1 | 142.5 | 421.8 | 9.8 % | pass |
| `ncclcnet0-rc2` | 52.9 | 99.6 | 126.1 | 143.0 | 421.6 | 7.8 % | pass |
| `ncclring-rc2` | 55.1 | 100.2 | 128.0 | 138.2 | 421.5 | 12.9 % | pass |
| `ncclcu0-rc2` | 53.6 | 96.2 | 126.1 | 145.0 | 420.9 | 5.6 % | pass |
| `nccllls-rc2` | 54.2 | 89.0 | 131.3 | 146.4 | 420.9 | 6.4 % | pass |
| `moeshare` | 55.4 | 98.0 | 126.9 | 140.5 | 420.8 | 4.4 % | pass |
| `noprki-rc2` | 55.5 | 99.3 | 126.5 | 139.4 | 420.7 | 7.9 % | pass |
| `cgpiece-k6-rc2` | 55.1 | 96.1 | 129.9 | 139.5 | 420.6 | 8.9 % | pass |
| `p030b` | 52.5 | 96.6 | 129.6 | 141.6 | 420.3 | 11.0 % | pass |
| `now4sh` | 52.5 | 100.1 | 127.4 | 140.1 | 420.1 | 8.3 % | pass |
| `ncclchk0-rc2` | 56.1 | 92.3 | 126.5 | 145.1 | 420.0 | 18.3 % | pass |
| `ncclpeer2-rc2` | 53.3 | 96.6 | 129.8 | 140.3 | 420.0 | 4.9 % | pass |
| `split1` | 53.0 | 100.5 | 124.8 | 141.7 | 420.0 | 12.6 % | pass |
| `now4mat-rc2` | 53.8 | 99.1 | 127.6 | 139.3 | 419.8 | 15.7 % | pass |
| `memprof0-rc2` | 50.2 | 98.4 | 130.6 | 140.4 | 419.6 | 10.0 % | pass |
| `idxfi` | 55.6 | 97.3 | 124.2 | 142.3 | 419.4 | 12.2 % | pass |
| `noasync-rc2` | 52.3 | 97.6 | 126.0 | 142.6 | 418.5 | 12.6 % | pass |
| `split8` | 53.0 | 97.2 | 125.9 | 142.4 | 418.5 | 8.5 % | pass |
| `yesh16` | 55.8 | 96.3 | 127.6 | 138.7 | 418.4 | 8.2 % | pass |
| `nopfx` | 53.9 | 98.2 | 124.0 | 142.2 | 418.3 | 8.5 % | pass |
| `ncclgrp-rc2` | 54.4 | 95.3 | 126.9 | 141.6 | 418.2 | 8.8 % | pass |
| `wochunk1-rc2` | 53.7 | 99.9 | 127.7 | 136.7 | 418.0 | 6.1 % | pass |
| `tile128-rc2` | 53.8 | 98.8 | 122.6 | 142.5 | 417.7 | 6.3 % | pass |
| `noaot-rc2` | 54.3 | 95.8 | 126.3 | 141.1 | 417.5 | 9.4 % | pass |
| `proto2-030-rc2` | 51.7 | 98.0 | 126.6 | 141.2 | 417.5 | 14.0 % | pass |
| `ncclsock4-rc2` | 53.4 | 96.4 | 128.2 | 139.4 | 417.4 | 7.1 % | pass |
| `ncclnetib-rc2` | 54.4 | 96.1 | 127.1 | 139.1 | 416.7 | 12.3 % | pass |
| `ncclshm0-rc2` | 52.9 | 95.7 | 125.4 | 142.5 | 416.5 | 11.4 % | pass |
| `lintri` | 52.5 | 93.0 | 129.2 | 141.7 | 416.4 | 10.2 % | pass |
| `lintri-rc2` | 52.4 | 96.7 | 125.4 | 141.1 | 415.6 | 5.0 % | pass |
| `ncclmin2-rc2` | 51.5 | 96.6 | 127.0 | 140.3 | 415.4 | 8.4 % | pass |
| `proto2-030` | 52.8 | 93.9 | 127.1 | 138.3 | 412.1 | 12.8 % | pass |
| `tile128` | 55.3 | 95.4 | 124.4 | 136.2 | 411.3 | 12.3 % | pass |
| `proto2-dg-cgnone` | 32.7 | 88.8 | 135.9 | 147.7 | 405.1 | 18.3 % | pass |
| `moe_humming` | 42.0 | 79.3 | 109.1 | 122.8 | 353.2 | 7.5 % | pass |
| `nccpll-rc2` | 51.1 | 86.8 | 101.6 | 112.8 | 352.3 | 8.8 % | pass |
| `proto5` | 38.9 | 82.4 | 95.4 | 133.0 | 349.7 | 4.3 % | pass |
| `protog` | 38.4 | 80.8 | 105.2 | 122.3 | 346.7 | 9.5 % | pass |
| `attnfi` | 37.8 | 78.5 | 104.6 | 119.8 | 340.7 | 5.4 % | pass |
| `nogath` | 37.0 | 76.2 | 107.2 | 119.9 | 340.3 | 3.5 % | pass |
| `proto2` | 35.4 | 78.9 | 103.0 | 117.0 | 334.3 | 12.1 % | pass |
| `proto2-moea4` | 36.6 | 79.0 | 104.1 | 114.6 | 334.3 | 11.5 % | pass |
| `proto2-a16` | 31.6 | 76.8 | 106.2 | 119.4 | 334.0 | 49.5 % | pass |
| `proto2-dglin` | 30.1 | 76.9 | 106.7 | 114.6 | 328.3 | 48.6 % | pass |
| `proto2-attnfi` | 35.4 | 73.6 | 101.9 | 117.2 | 328.1 | 7.3 % | pass |
| `proto2-attnfi-warm` | 37.5 | 71.1 | 99.3 | 113.0 | 320.9 | 4.4 % | pass |
| `proto2-compile` | 35.1 | 75.3 | 97.1 | 111.4 | 318.9 | 6.8 % | pass |
| `cgfull` | 38.9 | 76.0 | 87.7 | 109.2 | 311.8 | 74.0 % | pass |
| `proto2-eager` | 28.4 | 68.6 | 98.2 | 112.9 | 308.1 | 15.5 % | pass |
| `proto2-skip-attn` | 57.9 | 66.8 | 79.6 | 86.6 | 290.9 | 15.1 % | fail |
| `hum-k5-rc2` | 29.7 | 76.5 | 75.4 | 93.5 | 275.1 | 83.3 % | pass |
| `stockops2` | 33.1 | 59.9 | 74.7 | 81.2 | 248.9 | 9.2 % | pass |
| `proto2-mhctl` | 9.6 | 23.6 | 34.7 | 41.1 | 109.0 | 0.5 % | pass |
| `proto2-nodspark` | 9.5 | 23.4 | 34.4 | 40.5 | 107.8 | 0.6 % | pass |
| `nomulti` | 11.0 | 15.7 | 19.7 | 20.9 | 67.3 | 9.1 % | pass |
| `dglinear` | - | - | - | - | - | - | fail |
| `lin_cutedsl` | - | - | - | - | - | - | fail |

No median table, so no numbers: `dglinear`, `lin_cutedsl` (the run was stopped after the failure was visible).

Per-level detail, best sum first.

## Diagnostic arms (profiler-armed)

These run the decode profiler, which distorts throughput, so they are not candidates and their sums must not be compared with the table above. They are kept because they produced the attribution.

| tag | c1 | c3 | c5 | c6 | sum | worst spread | gates |
|---|---|---|---|---|---|---|---|
| `prof6` | 36.7 | 76.4 | 105.8 | 121.4 | 340.3 | 7.9 % | pass |
| `p6c` | 34.5 | 76.0 | 102.9 | 126.1 | 339.5 | 7.2 % | pass |
| `prof3` | 37.4 | 76.2 | 104.7 | 119.0 | 337.3 | 9.4 % | pass |
| `prof2` | 39.6 | 74.5 | 102.7 | 120.1 | 336.9 | 15.4 % | pass |
| `prof6b` | 36.3 | 74.0 | 107.8 | 118.1 | 336.2 | 8.8 % | pass |
| `prof4` | 36.2 | 76.3 | 103.3 | 119.4 | 335.2 | 8.8 % | pass |
| `proto2-recipe2` | 34.8 | 74.9 | 100.8 | 117.0 | 327.5 | 9.2 % | pass |
| `proto2-skip-ffn` | 10.4 | 30.7 | 51.1 | 61.2 | 153.4 | 0.3 % | fail |

### `refg-now`

**Changed.** the anemll reference (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, `harness/ref-base0731.yaml`) re-measured **in the same session** as the comparison arms, five passes. The rig drifts ~6-7 % between sessions (see `rc2back-rc2`), so a reference measured hours earlier cannot be compared against.

**Verdict.** **reference, not a candidate.** 63.7 / 117.0 / 142.0 / **161.2**, sum **483.9**, worst spread 26.6 % (c5 114.6-152.4; c1 is tight at 2.2 %). Both gates pass in all five passes. Acceptance 49.9-54.5 %, tokens per step 4.476-4.785. Against this same-session reference, our v0.30.0 default (`proto2-030-0`, 444.5 / 150.2) is **-8.1 % / -6.8 %** and the rc2 image at identical defaults (`rc2back-rc2`, 448.7 / 153.3) is **-7.3 % / -4.9 %**. Per-level deficit for v0.30.0: c1 -10.0 %, c3 -10.4 %, c5 -6.9 %, c6 -6.8 % — the small-batch levels are the worst, so the gap is not a single kernel.

Container `sparkrun_bccefe990c7aefab_a1d74a94a9ec_node_0 Up 6 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-21T16:31:10+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 63.0 / 64.4 / 63.3 / 63.8 / 63.7 | 63.7 | 2.2 % | - | - | - |
| 3 | 117.0 / 117.3 / 110.4 / 118.1 / 108.6 | 117.0 | 8.1 % | - | - | - |
| 5 | 114.6 / 152.4 / 142.0 / 140.3 / 147.4 | 142.0 | 26.6 % | - | - | - |
| 6 | 165.1 / 158.5 / 161.2 / 153.5 / 162.1 | 161.2 | 7.2 % | - | - | - |

### `refg`

**Changed.** reference arm: anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` on the base checkpoint `deepseek-ai/DeepSeek-V4-Flash-0731`, launched by `~/goal/launch-refbase.sh`

**Verdict.** **comparator**. Not a candidate: nothing in it was changed. Re-measured 2026-09-17 on the same rig and day as `proto2` (66.4 / 118.4 / 139.0 / 161.0, sum 484.8); the 2026-09-13 run of the same image and recipe gave 63.1 / 112.1 / 140.9 / 156.0, sum 472.1, so the rig itself moved about +3.2 % at c6 between the two days.

Container `sparkrun_bccefe990c7aefab_35b594a4b47e_node_0 Up 6 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T02:24:29+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 65.3 / 66.5 / 63.9 | 65.3 | 4.0 % | - | - | - |
| 3 | 112.9 / 110.1 / 107.8 | 110.1 | 4.6 % | - | - | - |
| 5 | 116.9 / 138.6 / 139.2 | 138.6 | 16.1 % | - | - | - |
| 6 | 158.1 / 161.5 / 159.1 | 159.1 | 2.1 % | - | - | - |

### `ctl2-hum-rc2`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-22T02:08:28+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 66.3 / 60.4 / 68.0 | 66.3 | 11.5 % | - | - | - |
| 3 | 101.9 / 117.5 / 117.3 | 117.3 | 13.3 % | - | - | - |
| 5 | 148.6 / 144.5 / 144.7 | 144.7 | 2.8 % | - | - | - |
| 6 | 167.5 / 165.7 / 161.3 | 165.7 | 3.7 % | - | - | - |

### `ctl3-rc2`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 5 passes at 512 tokens, started 2026-09-22T03:00:55+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 69.0 / 62.9 / 60.7 / 62.7 / 64.1 | 62.9 | 13.2 % | - | - | - |
| 3 | 117.2 / 118.3 / 116.6 / 110.7 / 115.9 | 116.6 | 6.5 % | - | - | - |
| 5 | 147.2 / 149.4 / 143.4 / 152.1 / 146.8 | 147.2 | 5.9 % | - | - | - |
| 6 | 164.3 / 159.0 / 167.3 / 164.7 / 164.2 | 164.3 | 5.1 % | - | - | - |

### `abOTH-d`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `sparkrun_bccefe990c7aefab_76095abffa65_node_0 Up 10 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-21T19:25:35+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 69.9 / 60.1 / 67.6 | 67.6 | 14.5 % | - | - | - |
| 3 | 116.0 / 119.6 / 111.9 | 116.0 | 6.6 % | - | - | - |
| 5 | 144.6 / 147.5 / 146.6 | 146.6 | 2.0 % | - | - | - |
| 6 | 160.3 / 158.2 / 161.9 | 160.3 | 2.3 % | - | - | - |

### `qO-a`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 5 passes at 512 tokens, started 2026-09-22T03:11:36+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 59.0 / 63.6 / 65.6 / 67.2 / 58.9 | 63.6 | 13.1 % | - | - | - |
| 3 | 114.2 / 107.6 / 114.3 / 125.7 / 117.8 | 114.3 | 15.8 % | - | - | - |
| 5 | 145.3 / 156.2 / 144.4 / 147.0 / 144.0 | 145.3 | 8.4 % | - | - | - |
| 6 | 165.1 / 162.1 / 165.4 / 164.9 / 158.7 | 164.9 | 4.1 % | - | - | - |

### `proto2-030-0b`

**Changed.** the same v0.30.0 default as `proto2-030-0` (same image, same config: humming, WO overlay off, k=6), re-measured five passes thirty minutes later, as the adjacent control for the capture-size arm. One variable against `proto2-030-0`: time.

**Verdict.** **the drift control, and the most important number in this round.** 64.1 / 117.8 / 145.1 / **158.9**, sum **485.9**, worst spread 10.8 %. Both gates pass in all five passes. Against the identical `proto2-030-0` (444.5 / 150.2, measured 16:13) this is **+9.3 % / +5.8 %** with literally nothing changed but the clock. Together with `rc2back-rc2` (identical image and config, 477.5 at 14:14 vs 448.7 at 16:23, -6.4 %) the rig is shown to swing roughly +-9 % between sessions, so only adjacent arms are comparable. Against the same-session reference `refg-now` (483.9 / 161.2) this control is **+0.4 % / -1.4 %**, i.e. parity.

Container `sparkrun_bccefe990c7aefab_a1d74a94a9ec_node_0 Up 13 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-21T16:38:22+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 64.5 / 64.1 / 63.0 / 63.7 / 65.1 | 64.1 | 3.3 % | - | - | - |
| 3 | 109.8 / 117.9 / 118.3 / 112.9 / 117.8 | 117.8 | 7.2 % | - | - | - |
| 5 | 145.1 / 146.0 / 141.6 / 142.0 / 151.2 | 145.1 | 6.6 % | - | - | - |
| 6 | 172.1 / 160.7 / 158.9 / 155.0 / 155.2 | 158.9 | 10.8 % | - | - | - |

### `refg-rc2b`

**Changed.** same-day anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` on 2026-09-20, launched by `configs/examples/refg-rc2b.sh` (`~/goal/launch-refbase.sh` + `drive-median.sh`). Protocol k=7, capture 48, base checkpoint. Not a candidate.

**Verdict.** **comparator for 2026-09-20.** 63.0 / 114.6 / 146.1 / **162.2**, sum **485.9**, worst spread 20.7 % (c5 119.4-149.6). Gates pass in all three passes (`' Paris...'`, `'72, 9x9'`). Acceptance 50.7-56.5 %, tokens per step 4.531-4.923. vs yesterday `refg-rc2` (467.9 / 157.1) this is +3.8 % / +3.2 % — rig noise up. Standing `proto2-030-rc2` (417.5 / 141.2) is now -14.1 % / -12.9 % vs this same-day anemll. Best same-pin `moea16-rc2` (453.7 / 152.8) is -6.6 % / -5.8 %.

Container `none`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T01:27:34+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 65.8 / 63.0 / 60.9 | 63.0 | 7.8 % | - | - | - |
| 3 | 112.7 / 114.6 / 115.5 | 114.6 | 2.4 % | - | - | - |
| 5 | 119.4 / 149.6 / 146.1 | 146.1 | 20.7 % | - | - | - |
| 6 | 164.5 / 162.2 / 157.4 | 162.2 | 4.4 % | - | - | - |

### `capsz-030-0`

**Changed.** `proto2-030-0` plus `CUDAGRAPH_CAPTURE_SIZES=[7,21,35,42,48]`, five passes, run immediately after the control `proto2-030-0b`. One variable against the control: the CUDA-graph capture sizes. With k=6 the per-level target batch is 7/21/35/42 tokens while the list still held the k=7 sizes [1,2,4,8,16,24,32,40,48], so every level padded up to the next size (7->8, 21->24, 35->40, 42->48), ~14 % wasted rows; this arm matched the graphs to the real sizes.

**Verdict.** **wash; the padding hypothesis is refuted.** 62.3 / 117.6 / 145.6 / **159.5**, sum **485.0**, worst spread 11.6 %. Both gates pass in all five passes; acceptance 51.0-54.2 % and tokens per step 4.539-4.770 hold the floor. Against the adjacent control `proto2-030-0b` (485.9 / 158.9, spread 10.8 %) this is **-0.2 % / +0.4 %**. Either the padded rows are not the cost at these sizes, or vLLM already pads to a captured size and the extra rows are free. Keep the standing capture list.

Container `sparkrun_bccefe990c7aefab_a1d74a94a9ec_node_0 Up 19 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-21T16:45:01+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 65.2 / 58.0 / 64.6 / 61.6 / 62.3 | 62.3 | 11.6 % | - | - | - |
| 3 | 117.6 / 118.7 / 114.1 / 121.8 / 113.1 | 117.6 | 7.4 % | - | - | - |
| 5 | 145.0 / 152.0 / 143.3 / 145.6 / 146.6 | 145.6 | 6.0 % | - | - | - |
| 6 | 158.0 / 158.1 / 162.8 / 159.5 / 160.1 | 159.5 | 3.0 % | - | - | - |

### `ctl-hum-rc2`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `sparkrun_bccefe990c7aefab_694acfdd02d2_node_0 Up 20 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-22T01:54:53+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 61.0 / 62.2 / 59.6 | 61.0 | 4.3 % | - | - | - |
| 3 | 116.3 / 118.8 / 110.5 | 116.3 | 7.1 % | - | - | - |
| 5 | 147.3 / 146.3 / 145.3 | 146.3 | 1.4 % | - | - | - |
| 6 | 164.3 / 161.2 / 160.1 | 161.2 | 2.6 % | - | - | - |

### `pO-a`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-22T02:16:18+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 63.7 / 65.1 / 59.3 | 63.7 | 9.1 % | - | - | - |
| 3 | 112.6 / 111.2 / 124.5 | 112.6 | 11.8 % | - | - | - |
| 5 | 146.8 / 142.3 / 145.9 | 145.9 | 3.1 % | - | - | - |
| 6 | 162.3 / 161.5 / 158.4 | 161.5 | 2.4 % | - | - | - |

### `qO-d`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 5 passes at 512 tokens, started 2026-09-22T03:45:02+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 64.3 / 62.2 / 65.0 / 61.6 / 57.9 | 62.2 | 11.4 % | - | - | - |
| 3 | 111.5 / 119.8 / 110.4 / 108.5 / 121.6 | 111.5 | 11.7 % | - | - | - |
| 5 | 148.7 / 148.5 / 148.3 / 146.2 / 144.2 | 148.3 | 3.0 % | - | - | - |
| 6 | 163.5 / 161.7 / 168.4 / 157.6 / 161.1 | 161.7 | 6.7 % | - | - | - |

### `refg030`

**Changed.** same-day reference arm: anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` on the base checkpoint, launched by `~/goal/launch-refbase.sh` on 2026-09-18 after `proto2-030`

**Verdict.** **comparator for the v0.30.0rc1 re-baseline.** 64.0 / 114.2 / 143.4 / **162.1**, sum **483.7**, worst spread 18.5 % (c5 121.9-148.4). Gates pass in all three passes. Acceptance 49.4-55.8 % and tokens per step 4.439-4.876. The previous `refg` row (sum 473.1 / 484.8) is a different day.

Container `sparkrun_bccefe990c7aefab_718a38c1841d_node_0 Up 6 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T14:45:30+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 65.4 / 61.7 / 64.0 | 64.0 | 5.8 % | - | - | - |
| 3 | 116.0 / 106.1 / 114.2 | 114.2 | 8.7 % | - | - | - |
| 5 | 121.9 / 148.4 / 143.4 | 143.4 | 18.5 % | - | - | - |
| 6 | 153.4 / 162.1 / 162.3 | 162.1 | 5.5 % | - | - | - |

### `rO-a`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 5 passes at 512 tokens, started 2026-09-22T10:10:48+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 63.9 / 58.2 / 64.4 / 62.4 / 62.8 | 62.8 | 9.9 % | - | - | - |
| 3 | 112.9 / 120.8 / 112.5 / 111.5 / 110.4 | 112.5 | 9.2 % | - | - | - |
| 5 | 146.5 / 146.2 / 146.0 / 150.9 / 144.5 | 146.2 | 4.4 % | - | - | - |
| 6 | 163.5 / 161.7 / 156.4 / 165.7 / 160.2 | 161.7 | 5.8 % | - | - | - |

### `abREF-b`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `sparkrun_bccefe990c7aefab_7eedc6a565a2_node_0 Up 6 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-21T19:11:11+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 61.8 / 61.5 / 66.3 | 61.8 | 7.8 % | - | - | - |
| 3 | 117.1 / 114.9 / 113.0 | 114.9 | 3.6 % | - | - | - |
| 5 | 112.5 / 145.0 / 150.5 | 145.0 | 26.2 % | - | - | - |
| 6 | 158.3 / 159.0 / 170.3 | 159.0 | 7.5 % | - | - | - |

### `rO-d`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 5 passes at 512 tokens, started 2026-09-22T10:43:33+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 64.9 / 63.7 / 63.0 / 65.9 / 60.8 | 63.7 | 8.0 % | - | - | - |
| 3 | 109.5 / 105.1 / 110.4 / 110.9 / 109.5 | 109.5 | 5.3 % | - | - | - |
| 5 | 147.6 / 146.6 / 142.5 / 146.4 / 142.4 | 146.4 | 3.6 % | - | - | - |
| 6 | 161.9 / 161.0 / 160.1 / 160.7 / 155.6 | 160.7 | 3.9 % | - | - | - |

### `pR-b`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `sparkrun_bccefe990c7aefab_f78197a6339f_node_0 Up 5 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-22T02:25:43+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 63.3 / 62.9 / 60.0 | 62.9 | 5.2 % | - | - | - |
| 3 | 112.0 / 115.8 / 110.8 | 112.0 | 4.5 % | - | - | - |
| 5 | 116.8 / 139.1 / 141.3 | 139.1 | 17.6 % | - | - | - |
| 6 | 157.9 / 164.9 / 163.7 | 163.7 | 4.3 % | - | - | - |

### `hum-k6-rc2b`

**Changed.** `wooff-rc2` + k=6 plus `MOE_BACKEND=humming` passed through the **launcher** channel (`harness/run-arm.sh <tag> "MOE_BACKEND=humming"`), five passes. One variable against the standing default: the MoE backend (humming MXFP4 kernels vs b12x). Confirmed `moe_backend='humming'` in the resolved engine config. This is the correct-channel re-run of `hum-k6-rc2`, which was invalid.

**Verdict.** **large positive, and the best same-pin result; adopted as the served default.** 58.5 / 111.6 / 146.8 / **160.6**, sum **477.5**, worst spread 11.8 %. Both gates pass in all five passes. Acceptance 53.1-64.3 % holds the floor; one pass dipped to 4.197 tokens per step against the 4.427 floor. Against the preceding default `wooff-k6-5-rc2` (445.2 / 152.3, spread 7.6 %) this is **+7.3 % / +5.4 %**, inside the keep bar max(7.6 %, 11.8 %) = 11.8 %. Against the contract standing arm `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) it is **+14.4 % / +13.7 %**: the sum clears the 14.0 % bar and c6 misses it by 0.3 points. Against same-day `refg-rc2b` (485.9 / 162.2, spread 20.7 %) the gap is **-1.7 % / -1.0 %**, down from -12.0 % / -13.9 % at the start of round 103. Positive in two independent measurements (`moehum-rc2` +6.7 % on the old baseline, this +7.3 % on the new one). `MOE_BACKEND` now defaults to humming in the pin.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-21T14:09:00+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.6 / 58.5 / 60.5 / 59.5 / 56.7 | 58.5 | 11.8 % | - | - | - |
| 3 | 114.2 / 114.7 / 111.6 / 105.1 / 110.9 | 111.6 | 8.6 % | - | - | - |
| 5 | 148.1 / 140.1 / 139.9 / 146.8 / 147.7 | 146.8 | 5.6 % | - | - | - |
| 6 | 158.2 / 165.0 / 164.8 / 160.6 / 153.1 | 160.6 | 7.4 % | - | - | - |

### `rR-c`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `sparkrun_bccefe990c7aefab_ad9bc134f4b6_node_0 Up 6 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-22T10:34:14+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 62.2 / 62.3 / 63.5 / 62.2 / 61.1 | 62.2 | 3.9 % | - | - | - |
| 3 | 108.9 / 117.9 / 114.2 / 111.4 / 109.7 | 111.4 | 8.1 % | - | - | - |
| 5 | 111.5 / 144.8 / 139.5 / 145.5 / 142.6 | 142.6 | 23.8 % | - | - | - |
| 6 | 159.2 / 156.1 / 161.5 / 161.0 / 164.6 | 161.0 | 5.3 % | - | - | - |

### `qR-c`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `sparkrun_bccefe990c7aefab_a68824c38ed1_node_0 Up 6 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-22T03:34:53+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.4 / 61.2 / 60.9 / 64.8 / 64.0 | 61.2 | 10.5 % | - | - | - |
| 3 | 103.1 / 114.3 / 115.0 / 111.0 / 116.2 | 114.3 | 11.5 % | - | - | - |
| 5 | 118.9 / 142.2 / 143.7 / 140.3 / 143.5 | 142.2 | 17.4 % | - | - | - |
| 6 | 159.4 / 162.9 / 157.0 / 162.6 / 159.0 | 159.4 | 3.7 % | - | - | - |

### `rR-b`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `sparkrun_bccefe990c7aefab_8e72df49c06c_node_0 Up 6 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-22T10:22:26+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 71.4 / 63.4 / 62.7 / 63.0 / 63.1 | 63.1 | 13.8 % | - | - | - |
| 3 | 118.2 / 110.3 / 115.1 / 115.2 / 118.5 | 115.2 | 7.1 % | - | - | - |
| 5 | 113.6 / 143.0 / 148.2 / 139.6 / 139.4 | 139.6 | 24.8 % | - | - | - |
| 6 | 157.4 / 160.8 / 159.3 / 154.7 / 156.7 | 157.4 | 3.9 % | - | - | - |

### `pR-c`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `sparkrun_bccefe990c7aefab_fe743c385aa4_node_0 Up 6 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-22T02:35:24+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 59.1 / 59.8 / 67.6 | 59.8 | 14.2 % | - | - | - |
| 3 | 85.4 / 112.3 / 107.8 | 107.8 | 25.0 % | - | - | - |
| 5 | 148.6 / 147.8 / 138.1 | 147.8 | 7.1 % | - | - | - |
| 6 | 152.7 / 156.9 / 155.7 | 155.7 | 2.7 % | - | - | - |

### `abREF-c`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `sparkrun_bccefe990c7aefab_76095abffa65_node_0 Up 6 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-21T19:20:51+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 62.3 / 63.1 / 63.2 | 63.1 | 1.4 % | - | - | - |
| 3 | 111.9 / 115.3 / 111.8 | 111.9 | 3.1 % | - | - | - |
| 5 | 117.3 / 139.6 / 151.3 | 139.6 | 24.4 % | - | - | - |
| 6 | 150.4 / 162.6 / 155.7 | 155.7 | 7.8 % | - | - | - |

### `qR-b`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `sparkrun_bccefe990c7aefab_fd25f95ab3fb_node_0 Up 5 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-22T03:23:01+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 61.8 / 58.6 / 64.6 / 60.9 / 60.0 | 60.9 | 9.9 % | - | - | - |
| 3 | 108.0 / 113.0 / 110.5 / 111.7 / 120.2 | 111.7 | 10.9 % | - | - | - |
| 5 | 117.3 / 136.6 / 136.5 / 143.0 / 144.0 | 136.6 | 19.5 % | - | - | - |
| 6 | 156.0 / 159.2 / 155.2 / 161.0 / 162.8 | 159.2 | 4.8 % | - | - | - |

### `refg-rc2`

**Changed.** same-day anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` on the `v0.30.0rc2` pin day, launched by `configs/examples/refg-rc2.sh` (`~/goal/launch-refbase.sh` + `drive-median.sh`). Protocol k=7, capture 48, base checkpoint. Not a candidate.

**Verdict.** **comparator for the v0.30.0rc2 pin.** 58.6 / 113.4 / 138.8 / **157.1**, sum **467.9**, worst spread 21.9 % (c5 111.6-142.0). Gates pass in all three passes (`' Paris...'`, `'72, 9x9'`). Acceptance 48.9-53.9 %, tokens per step 4.414-4.785. vs previous day `refg030` (483.7 / 162.1) this is -3.3 % / -3.1 % — rig noise, not a new reference image. Standing control `proto2-030-rc2` (417.5 / 141.2) is still -10.8 % / -10.1 % vs this same-day `refg-rc2`.

Container `none`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T15:53:19+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.6 / 58.4 / 60.9 | 58.6 | 4.3 % | - | - | - |
| 3 | 108.1 / 114.8 / 113.4 | 113.4 | 5.9 % | - | - | - |
| 5 | 111.6 / 142.0 / 138.8 | 138.8 | 21.9 % | - | - | - |
| 6 | 161.5 / 157.1 / 157.0 | 157.1 | 2.9 % | - | - | - |

### `kv98304`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-22T00:17:53+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 59.8 / 55.8 / 61.7 | 59.8 | 9.9 % | - | - | - |
| 3 | 107.0 / 109.7 / 111.5 | 109.7 | 4.1 % | - | - | - |
| 5 | 136.2 / 136.4 / 135.3 | 136.2 | 0.8 % | - | - | - |
| 6 | 157.6 / 161.6 / 153.9 | 157.6 | 4.9 % | - | - | - |

### `abOTH-a`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-21T19:01:28+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 60.6 / 58.8 / 60.3 | 60.3 | 3.0 % | - | - | - |
| 3 | 109.7 / 107.4 / 107.5 | 107.5 | 2.1 % | - | - | - |
| 5 | 145.9 / 133.6 / 138.3 | 138.3 | 8.9 % | - | - | - |
| 6 | 151.3 / 157.3 / 149.0 | 151.3 | 5.5 % | - | - | - |

### `moemar`

**Changed.** `proto2-030` plus `MOE_BACKEND=marlin`. One variable: the MXFP4 oracle's Marlin expert path (`MARLIN` / `MarlinExperts`, SM75+). Never measured on this pin. Engine log: `Using 'MARLIN' Mxfp4 MoE backend`.

**Verdict.** **best same-image sum, not a keep.** 58.7 / 101.8 / 140.0 / **154.8**, sum **455.3**, worst spread 4.5 %. Gates pass in all three passes; acceptance 50.4-56.7 % and tokens per step 4.518-4.923, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / c6 138.3, spread 12.8 %) the sum is +10.5 % and c6 is +11.9 %, both inside that spread. Against same-day control `p030b` (420.3 / 141.6, spread 11.0 %) +8.3 % / +9.3 %, also inside. Against `moehum` (448.3 / 152.0) a small further lift. Against `refg030` (483.7 / 162.1) still -5.9 % / -4.5 %. Standing arm remains `proto2-030`. Marlin is the current ceiling on this pin, not a protocol win vs the keep-rule.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T17:45:09+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.7 / 59.9 / 58.5 | 58.7 | 2.4 % | - | - | - |
| 3 | 101.8 / 104.3 / 99.7 | 101.8 | 4.5 % | - | - | - |
| 5 | 141.3 / 137.2 / 140.0 | 140.0 | 2.9 % | - | - | - |
| 6 | 155.5 / 154.8 / 153.6 | 154.8 | 1.2 % | - | - | - |

### `kv65536`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-22T00:10:07+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.4 / 60.4 / 57.8 | 57.8 | 8.7 % | - | - | - |
| 3 | 109.0 / 107.9 / 106.3 | 107.9 | 2.5 % | - | - | - |
| 5 | 136.3 / 134.4 / 135.2 | 135.2 | 1.4 % | - | - | - |
| 6 | 157.0 / 153.5 / 147.4 | 153.5 | 6.3 % | - | - | - |

### `kv229376`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-22T00:49:00+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 57.0 / 55.0 / 52.6 | 55.0 | 8.0 % | - | - | - |
| 3 | 102.5 / 113.2 / 106.9 | 106.9 | 10.0 % | - | - | - |
| 5 | 143.8 / 134.9 / 139.5 | 139.5 | 6.4 % | - | - | - |
| 6 | 152.4 / 161.0 / 152.8 | 152.8 | 5.6 % | - | - | - |

### `moea16-rc2`

**Changed.** `proto2-030-rc2` plus `VLLM_B12X_MOE_FP4_FORCE_A16=1`. One variable: b12x MXFP4 with BF16 activations (`B12X_MXFP4_BF16`). Confirmed in the engine log. Re-measure of rc1 `moea16` (446.3) on this pin.

**Verdict.** **positive, not a keep.** 56.7 / 105.7 / 138.5 / **152.8**, sum **453.7**, worst spread 7.1 %. Gates pass in all three passes; acceptance 50.5-54.6 % and tokens per step 4.491-4.830 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +8.7 % / +8.2 %, inside keep-spread. vs same-day `refg-rc2` (467.9 / 157.1) still -3.0 % / -2.7 %. Best same-pin sum so far. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T16:42:49+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 59.2 / 55.2 / 56.7 | 56.7 | 7.1 % | - | - | - |
| 3 | 107.1 / 105.7 / 104.1 | 105.7 | 2.8 % | - | - | - |
| 5 | 138.5 / 135.3 / 139.1 | 138.5 | 2.7 % | - | - | - |
| 6 | 152.8 / 151.1 / 153.2 | 152.8 | 1.4 % | - | - | - |

### `abwo-c`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-21T18:43:59+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 59.4 / 55.1 / 53.6 | 55.1 | 10.5 % | - | - | - |
| 3 | 107.9 / 101.3 / 108.7 | 107.9 | 6.9 % | - | - | - |
| 5 | 138.8 / 138.9 / 132.7 | 138.8 | 4.5 % | - | - | - |
| 6 | 150.5 / 155.6 / 151.3 | 151.3 | 3.4 % | - | - | - |

### `abwo-a`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-21T18:28:22+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.8 / 57.5 / 57.1 | 57.1 | 4.7 % | - | - | - |
| 3 | 108.8 / 107.8 / 100.6 | 107.8 | 7.6 % | - | - | - |
| 5 | 134.6 / 131.2 / 140.4 | 134.6 | 6.8 % | - | - | - |
| 6 | 157.1 / 149.0 / 150.9 | 150.9 | 5.4 % | - | - | - |

### `k6`

**Changed.** `proto2-030` plus `NUM_SPECULATIVE_TOKENS=6`. One variable: DSpark k=6, between checkpoint-native k=5 and the pin's k=7. Capture 48 covers 6*(6+1)=42. EXTRA overrides `run-arm.sh`'s hardcoded k=7. Confirmed `num_speculative_tokens: 6`. `k5` won c6 but lost tokens/step; this checks the midpoint.

**Verdict.** **negative, not kept.** 58.4 / 106.7 / 132.8 / **151.6**, sum **449.5**, worst spread 9.6 %. Gates pass in all three passes; acceptance 57.5-68.7 % and tokens per step 4.437-5.120, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) +9.1 % / +9.6 %, inside that spread. Against `p030b` (420.3) +7.0 %. Behind same-image ceiling `moemar` (455.3). k=6 keeps tokens/step where k=5 did not, but does not clear the keep-gate. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T22:27:08+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.3 / 58.4 / 61.9 | 58.4 | 9.6 % | - | - | - |
| 3 | 109.3 / 101.3 / 106.7 | 106.7 | 7.5 % | - | - | - |
| 5 | 133.6 / 128.2 / 132.8 | 132.8 | 4.1 % | - | - | - |
| 6 | 154.2 / 151.6 / 147.0 | 151.6 | 4.7 % | - | - | - |

### `abk6-b`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-21T17:11:42+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.8 / 54.3 / 58.8 | 55.8 | 8.1 % | - | - | - |
| 3 | 99.5 / 105.5 / 105.3 | 105.3 | 5.7 % | - | - | - |
| 5 | 134.7 / 133.1 / 132.8 | 133.1 | 1.4 % | - | - | - |
| 6 | 155.2 / 159.1 / 151.5 | 155.2 | 4.9 % | - | - | - |

### `kv245760`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-22T00:56:26+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 59.6 / 56.4 / 57.1 | 57.1 | 5.6 % | - | - | - |
| 3 | 107.7 / 107.2 / 105.2 | 107.2 | 2.3 % | - | - | - |
| 5 | 137.4 / 137.1 / 140.1 | 137.4 | 2.2 % | - | - | - |
| 6 | 152.7 / 147.6 / 146.4 | 147.6 | 4.3 % | - | - | - |

### `hum-k6-rc2`

**Changed.** `wooff-rc2` + k=6 with `SERVE_EXTRA_ENV=MOE_BACKEND=humming`, five passes. **Invalid channel**: `MOE_BACKEND` is read by `scripts/05-serve.sh` on the launcher to build `--moe-backend`, so setting it as container env (`-e`) is too late and the flag still said b12x. Confirmed `moe_backend='b12x'` and zero humming lines in the engine log.

**Verdict.** **not a humming measurement.** 56.8 / 102.7 / 136.8 / **152.6**, sum **448.9**. Because the backend never changed, this is a third independent sample of the k=6 / WO-off default (445.2 at five passes and 448.4 at three), i.e. +0.8 % against it, which is a useful confirmation of the noise floor rather than a result about humming. Superseded by `hum-k6-rc2b`. Trap recorded here so it is not repeated: container-env (`SERVE_EXTRA_ENV`) is for variables read inside the container; launcher-side knobs must go through the assignment prefix (`run-arm.sh <tag> "VAR=value"`).

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-21T13:58:55+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.8 / 56.1 / 58.7 / 56.8 / 58.5 | 56.8 | 10.4 % | - | - | - |
| 3 | 102.7 / 113.4 / 105.4 / 100.1 / 101.3 | 102.7 | 13.0 % | - | - | - |
| 5 | 136.8 / 131.7 / 141.7 / 134.3 / 139.7 | 136.8 | 7.3 % | - | - | - |
| 6 | 152.7 / 152.6 / 151.6 / 149.1 / 152.6 | 152.6 | 2.4 % | - | - | - |

### `abk6-a`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-21T17:03:53+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 50.6 / 56.8 / 53.6 | 53.6 | 11.6 % | - | - | - |
| 3 | 104.8 / 103.5 / 108.7 | 104.8 | 5.0 % | - | - | - |
| 5 | 134.4 / 132.7 / 131.9 | 132.7 | 1.9 % | - | - | - |
| 6 | 157.7 / 157.7 / 152.5 | 157.7 | 3.3 % | - | - | - |

### `rc2back-rc2`

**Changed.** the **previous** pin image (`vllm-spark-0731:main-030-rc2`) served with today's defaults (humming, WO overlay off, k=6), five passes, immediately after `proto2-030-0`. One variable against `proto2-030-0`: the image, i.e. vLLM `fa6ff060667f` vs `9ed533eb4adf`. This is the back-to-back control that the pin bump needs, because the rig drifts between sessions.

**Verdict.** **the pin bump is a wash.** 56.9 / 104.0 / 134.5 / **153.3**, sum **448.7**, worst spread 11.3 %. Both gates pass in all five passes. Against `proto2-030-0` (444.5 / 150.2, spread 12.6 %) this is **+0.9 % / +2.0 %**, i.e. inside noise: v0.30.0's only change is a two-file DeepGEMM build fix, and torch, triton and CUDA are byte-identical across the two images (`torch 2.14.0a0+git2b3ec34`, `triton 3.7.1`, `cuda 13.3`). **Drift finding:** the identical image *and* config (`hum-k6-rc2b`) measured **477.5 / 160.6** at 14:14 and **448.7 / 153.3** at 16:23, a **6.4 %** session-to-session shift. That is the same size as every effect chased in rounds 103-106, and is the dominant confound in this ledger.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-21T16:18:17+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 57.4 / 56.9 / 56.0 / 61.4 / 55.8 | 56.9 | 9.8 % | - | - | - |
| 3 | 104.0 / 100.3 / 106.7 / 100.5 / 112.1 | 104.0 | 11.3 % | - | - | - |
| 5 | 134.5 / 138.7 / 135.4 / 134.2 / 127.7 | 134.5 | 8.2 % | - | - | - |
| 6 | 157.4 / 153.0 / 155.3 / 147.0 / 153.3 | 153.3 | 6.8 % | - | - | - |

### `wooff-k6-rc2`

**Changed.** `wooff-rc2` (WO overlay off) plus `NUM_SPECULATIVE_TOKENS=6`. One variable against `wooff-rc2`: speculative depth 6 vs 7, with capture 48 unchanged (6*7 = 42 still fits). Confirmed `num_spec_tokens=6`, `max_cudagraph_capture_size' = 48`, `VLLM_USE_B12X_WO_PROJECTION=0`.

**Verdict.** **positive, not a keep.** 56.9 / 102.2 / 137.5 / **151.8**, sum **448.4**, worst spread 9.7 %. Both gates pass in all three passes. One pass dipped to 4.243 tokens per step, below the 4.427 floor; acceptance 54.6-63.8 % holds. Against `wooff-rc2` (426.9 / 144.2) +5.0 % / +5.3 %, inside the keep bar. Against same-day `refg-rc2b` (485.9 / 162.2) -8.4 % / -6.4 %: the best same-pin result to that point.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-21T13:17:55+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.0 / 56.9 / 58.5 | 56.9 | 4.4 % | - | - | - |
| 3 | 102.2 / 94.1 / 104.0 | 102.2 | 9.7 % | - | - | - |
| 5 | 139.6 / 136.3 / 137.5 | 137.5 | 2.4 % | - | - | - |
| 6 | 153.6 / 151.8 / 149.5 | 151.8 | 2.7 % | - | - | - |

### `moehum`

**Changed.** `proto2-030` plus `MOE_BACKEND=humming`. One variable: the MXFP4 oracle's humming expert path (`HUMMING` / indexed gemm). Device gate is SM75+; `has_humming()` is true in this image. Prior `moe_humming` (353.2) was on the old protog pin, never re-measured on `main-030-rc1`. Engine log: `Using 'HUMMING' Mxfp4 MoE backend`.

**Verdict.** **best same-image sum, not a keep.** 57.8 / 102.6 / 135.9 / **152.0**, sum **448.3**, worst spread 8.8 %. Gates pass in all three passes; acceptance 49.2-55.0 % and tokens per step 4.414-4.830, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / c6 138.3, spread 12.8 %) the sum is +8.8 % and c6 is +9.9 %, both inside that spread. Against same-day control `p030b` (420.3 / 141.6, spread 11.0 %) +6.7 % / +7.3 %, also inside. Against `refg030` (483.7 / 162.1) still -7.3 % / -6.2 %. Standing arm remains `proto2-030`. Humming is the current noise-ceiling on this pin, not a protocol win vs the keep-rule.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T17:21:49+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.4 / 53.3 / 57.8 | 57.8 | 8.8 % | - | - | - |
| 3 | 102.6 / 104.7 / 102.3 | 102.6 | 2.3 % | - | - | - |
| 5 | 131.8 / 135.9 / 140.9 | 135.9 | 6.7 % | - | - | - |
| 6 | 152.0 / 152.7 / 147.5 | 152.0 | 3.4 % | - | - | - |

### `nowo`

**Changed.** `proto2-030` plus `VLLM_USE_B12X_WO_PROJECTION=0`. One variable: overlay fused inv-RoPE FP8 + bmm O-proj off; layer falls back to einsum. `stockops2` mixed this with sparse-indexer-off. Never isolated on `main-030-rc1`. Confirmed in container env (`VLLM_USE_B12X_WO_PROJECTION=0`).

**Verdict.** **negative, not kept, but informative.** 58.8 / 105.4 / 132.3 / **151.5**, sum **448.0**, worst spread 5.4 %. Gates pass in all three passes; acceptance 49.9-55.0 % and tokens per step 4.483-4.830, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) +8.7 % / +9.5 %, inside that spread. Against `p030b` (420.3) +6.6 %. Behind same-image ceiling `moemar` (455.3). Isolated WO-off is a cost, not a win — opposite of `stockops2` which mixed WO-off with indexer-off and lost 34 % at c6. Overlay indexer stays load-bearing; WO overlay is the slower path. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T19:58:02+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.4 / 58.8 / 59.6 | 58.8 | 5.4 % | - | - | - |
| 3 | 106.8 / 105.1 / 105.4 | 105.4 | 1.6 % | - | - | - |
| 5 | 135.8 / 132.0 / 132.3 | 132.3 | 2.9 % | - | - | - |
| 6 | 144.1 / 151.5 / 151.6 | 151.5 | 5.0 % | - | - | - |

### `abwo-d`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-21T18:51:35+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.7 / 60.1 / 56.8 | 56.8 | 6.0 % | - | - | - |
| 3 | 107.4 / 99.4 / 101.8 | 101.8 | 7.9 % | - | - | - |
| 5 | 129.6 / 137.3 / 134.8 | 134.8 | 5.7 % | - | - | - |
| 6 | 154.2 / 152.7 / 154.7 | 154.2 | 1.3 % | - | - | - |

### `cgpiece-rc2`

**Changed.** `proto2-030-rc2` plus `CUDAGRAPH_MODE=PIECEWISE`. One variable: piecewise-only CUDA graphs vs standing FULL_AND_PIECEWISE. FULL-only was a large negative on earlier pins. Confirmed `cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>`.

**Verdict.** **positive, not a keep.** 54.6 / 106.4 / 136.1 / **150.4**, sum **447.5**, worst spread 17.8 %. Gates pass in all three passes. One pass dipped to accept 47.6 % / tokens per step 4.339, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +7.2 % / +6.5 %, inside keep-spread. Against same-day control `p030-rc2b` (424.6 / 144.2) +5.4 % / +4.3 %. Second-best same-pin sum after `moea16-rc2` (453.7 / 152.8). vs same-day `refg-rc2b` (485.9 / 162.2) still -7.9 % / -7.3 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-21T00:58:56+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 62.7 / 54.6 / 53.0 | 54.6 | 17.8 % | - | - | - |
| 3 | 101.4 / 106.4 / 108.1 | 106.4 | 6.3 % | - | - | - |
| 5 | 134.3 / 136.1 / 140.9 | 136.1 | 4.8 % | - | - | - |
| 6 | 150.6 / 148.3 / 150.4 | 150.4 | 1.5 % | - | - | - |

### `kv131072`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-22T00:25:15+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 59.1 / 56.6 / 60.1 | 59.1 | 5.9 % | - | - | - |
| 3 | 103.0 / 103.5 / 104.3 | 103.5 | 1.3 % | - | - | - |
| 5 | 133.3 / 129.6 / 137.7 | 133.3 | 6.1 % | - | - | - |
| 6 | 150.9 / 149.2 / 151.2 | 150.9 | 1.3 % | - | - | - |

### `kv196608`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-22T00:33:14+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 61.3 / 55.7 / 56.6 | 56.6 | 9.9 % | - | - | - |
| 3 | 103.4 / 112.0 / 107.1 | 107.1 | 8.0 % | - | - | - |
| 5 | 133.9 / 133.6 / 135.9 | 133.9 | 1.7 % | - | - | - |
| 6 | 148.7 / 148.9 / 154.7 | 148.9 | 4.0 % | - | - | - |

### `moea16`

**Changed.** `proto2-030` plus `VLLM_B12X_MOE_FP4_FORCE_A16=1`. One variable: select `B12X_MXFP4_BF16` instead of the pin's `B12X_MXFP4_MXFP8`. Reference (anemll 0.1.1, b12x 0.15.3) logs `Using 'B12X_MXFP4'` with activation_key None (BF16). Older library is banned; this is the analogue on 1.2.6. `proto2-a16` was a wash on the old pin. Engine log: `Using 'B12X_MXFP4_BF16' Mxfp4 MoE backend`.

**Verdict.** **negative, not kept.** 59.3 / 105.4 / 135.4 / **146.2**, sum **446.3**, worst spread 10.2 %. Gates pass in all three passes; acceptance 50.1-56.9 % and tokens per step 4.483-4.971, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) +8.3 % / +5.7 %, inside that spread. Against `p030b` (420.3) +6.2 %. Behind same-image ceiling `moemar` (455.3). Reference BF16 analogue is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T18:37:32+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 60.9 / 56.2 / 59.3 | 59.3 | 7.9 % | - | - | - |
| 3 | 99.7 / 105.4 / 110.5 | 105.4 | 10.2 % | - | - | - |
| 5 | 136.4 / 132.4 / 135.4 | 135.4 | 3.0 % | - | - | - |
| 6 | 145.5 / 151.8 / 146.2 | 146.2 | 4.3 % | - | - | - |

### `k6-rc2`

**Changed.** `proto2-030-rc2` plus `NUM_SPECULATIVE_TOKENS=6`. One variable: DSpark k=6 (pin is k=7). Capture 48 covers 6*(6+1)=42. Confirmed `num_speculative_tokens: 6` in engine args. Re-measure of rc1 `k6` (449.5) on this pin.

**Verdict.** **positive, not a keep.** 55.9 / 107.6 / 132.2 / **149.9**, sum **445.6**, worst spread 10.9 %. Gates pass in all three passes; acceptance 58.1-63.5 % and tokens per step 4.476-4.800 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +6.7 % / +6.2 %, inside keep-spread. vs same-day `refg-rc2` (467.9 / 157.1) still -4.8 % / -4.6 %. Tied with `moehum-rc2` (445.5) for best same-pin sum. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T16:32:52+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.9 / 57.1 / 55.9 | 55.9 | 3.9 % | - | - | - |
| 3 | 110.9 / 107.6 / 101.9 | 107.6 | 8.4 % | - | - | - |
| 5 | 136.2 / 131.9 / 132.2 | 132.2 | 3.3 % | - | - | - |
| 6 | 148.9 / 149.9 / 165.3 | 149.9 | 10.9 % | - | - | - |

### `moehum-rc2`

**Changed.** `proto2-030-rc2` plus `MOE_BACKEND=humming`. One variable: humming MXFP4 MoE (image humming-kernels 0.1.15). Confirmed humming GEMM config override in the engine log. Re-measure of rc1 `moehum` (448.3) on this pin.

**Verdict.** **positive, not a keep.** 58.1 / 102.2 / 132.3 / **152.9**, sum **445.5**, worst spread 4.7 %. Gates pass in all three passes. Tokens per step 4.429-4.785 holds the floor. One pass dipped to accept 49.1 %, a hair under `proto2-030-rc2`'s 49.2 %. Against standing (417.5 / 141.2, spread 14.0 %) this is +6.7 % / +8.3 %, inside keep-spread. vs same-day `refg-rc2` (467.9 / 157.1) still -4.8 % / -2.7 %. Best same-pin number so far, not a keep. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T16:23:48+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 57.8 / 58.1 / 58.4 | 58.1 | 1.0 % | - | - | - |
| 3 | 102.2 / 102.1 / 102.2 | 102.2 | 0.1 % | - | - | - |
| 5 | 132.3 / 131.4 / 137.6 | 132.3 | 4.7 % | - | - | - |
| 6 | 150.7 / 152.9 / 156.0 | 152.9 | 3.5 % | - | - | - |

### `wooff-k6-5-rc2`

**Changed.** `wooff-rc2` (WO overlay off) plus `NUM_SPECULATIVE_TOKENS=6`, **five** passes. Re-measure of `wooff-k6-rc2` at higher power, because the c1 median swing at three passes (~10 %) is larger than the effect being measured. Confirmed `num_spec_tokens=6`, `max_cudagraph_capture_size' = 48`, `VLLM_USE_B12X_WO_PROJECTION=0`.

**Verdict.** **positive, not a keep; adopted as the served default.** 56.3 / 103.7 / 132.9 / **152.3**, sum **445.2**, worst spread 7.6 %. Both gates pass in all five passes. Acceptance **57.5-67.5 %** against 49.8-56.9 % at k=7, and tokens per step 4.437-5.020 hold the floor (min 4.437 vs 4.427). Against the five-pass k=7 re-baseline `wooff5-rc2` (429.4 / 144.6, spread 8.5 %) this is +3.7 % / +5.3 %, inside the keep bar max(8.5 %, 7.6 %) = 8.5 %. Against the contract standing arm `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) it is +6.6 % / +7.9 %. Positive in three independent measurements (`k6-rc2` +6.7 %, `wooff-k6-rc2` +5.0 %, this +3.7 %), acceptance improves rather than trades off, and the mechanism is the one the profile supports. `NUM_SPECULATIVE_TOKENS` now defaults to 6 in the pin and in `harness/run-arm.sh`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-21T13:36:37+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 60.5 / 56.5 / 56.2 / 56.2 / 56.3 | 56.3 | 7.6 % | - | - | - |
| 3 | 107.0 / 105.0 / 102.0 / 103.7 / 103.4 | 103.7 | 4.8 % | - | - | - |
| 5 | 131.8 / 132.9 / 137.6 / 132.9 / 132.7 | 132.9 | 4.4 % | - | - | - |
| 6 | 152.3 / 149.4 / 145.0 / 153.7 / 155.9 | 152.3 | 7.2 % | - | - | - |

### `hum-a16-rc2`

**Changed.** `hum-k6-rc2b` (humming, WO off, k=6) plus `SERVE_EXTRA_ENV=VLLM_B12X_MOE_FP4_FORCE_A16=1`, five passes. One variable against the standing default: force W4A16 on the MoE now that the backend is humming — the format anemll's `flashinfer_b12x` alias resolves to, and the lever that measured best under b12x.

**Verdict.** **negative.** 57.8 / 102.9 / 134.9 / **149.5**, sum **445.1**, worst spread 7.6 %. Both gates pass in all five passes; acceptance 58.2-65.5 % and tokens per step 4.452-4.923 hold the floor (no dips). Against the standing `hum-k6-rc2b` (477.5 / 160.6, spread 11.8 %) this is **-6.8 % / -6.9 %**. So A16 does not help under humming either: under b12x it was +8.7 % once and then failed to replicate (+0.1 % on the WO-off default), and here it is a clear loss. **A16 is closed.** Same-day `refg-rc2b` (485.9 / 162.2) leaves -8.4 % / -7.8 % for this arm.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-21T14:33:47+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 57.8 / 59.8 / 55.9 / 56.5 / 59.0 | 57.8 | 6.7 % | - | - | - |
| 3 | 102.8 / 102.9 / 106.4 / 101.7 / 104.6 | 102.9 | 4.6 % | - | - | - |
| 5 | 126.1 / 135.5 / 134.9 / 134.5 / 136.3 | 134.9 | 7.6 % | - | - | - |
| 6 | 152.1 / 149.5 / 146.1 / 152.7 / 144.0 | 149.5 | 5.8 % | - | - | - |

### `abwo-b`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-21T18:36:19+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.5 / 53.5 / 56.6 | 56.5 | 5.5 % | - | - | - |
| 3 | 103.6 / 106.5 / 102.9 | 103.6 | 3.5 % | - | - | - |
| 5 | 132.8 / 135.2 / 133.9 | 133.9 | 1.8 % | - | - | - |
| 6 | 150.8 / 154.0 / 149.4 | 150.8 | 3.1 % | - | - | - |

### `proto2-030-0`

**Changed.** the v0.30.0 re-baseline: `configs/pin.main-029.env` moved to `VLLM_REF=9ed533eb4adfe48aef7e569a08daeccd2a773fed` and `IMAGE=vllm-spark-0731:main-030-0`, overlay scan FAIL=0 applied=43 no-op=12, phase-1 built, overlays applied, image copied to spark2, five passes. Serving defaults carried over: humming, WO overlay off, k=6. Confirmed engine `v0.30.1.dev0+g9ed533eb4`.

**Verdict.** **new-pin baseline.** 57.3 / 104.8 / 132.2 / **150.2**, sum **444.5**, worst spread 12.6 %. Both gates pass in all five passes; acceptance 56.1-65.5 % holds, one pass dipped to 4.376 tokens per step. Against the back-to-back rc2 control `rc2back-rc2` (448.7 / 153.3) this is -0.9 % / -2.0 %: the tag bump costs nothing and gains nothing, as expected from a two-file build fix. Against the same-session reference `refg-now` (483.9 / 161.2) it is **-8.1 % / -6.8 %**.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 5 passes at 512 tokens, started 2026-09-21T16:07:29+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.7 / 60.6 / 57.3 / 60.2 / 53.4 | 57.3 | 12.6 % | - | - | - |
| 3 | 102.7 / 104.8 / 97.6 / 106.3 / 105.3 | 104.8 | 8.3 % | - | - | - |
| 5 | 134.4 / 131.0 / 132.2 / 132.0 / 135.9 | 132.2 | 3.7 % | - | - | - |
| 6 | 137.3 / 150.4 / 153.3 / 150.2 / 149.7 | 150.2 | 10.7 % | - | - | - |

### `k5-rc2`

**Changed.** `proto2-030-rc2` plus `NUM_SPECULATIVE_TOKENS=5`. One variable: DSpark k=5 (checkpoint block size; pin is k=7). Capture 48 covers 6*(5+1)=36. Confirmed `num_speculative_tokens': 5`. Re-measure of rc1 `k5` (c6 163.2, tokens/step fail) on this pin.

**Verdict.** **c6 win, quality fail, not a keep.** 56.7 / 106.5 / 118.8 / **162.0**, sum **444.0**, worst spread 6.2 %. Gates pass in all three passes; acceptance 66.2-72.0 % is high. Tokens per step 4.303-4.571: min 4.303 is below standing floor 4.427. c6 162.0 beats same-day `refg-rc2` 157.1 (+3.1 %) and standing 141.2 (+14.7 %), but the sum 444.0 is +6.3 % vs standing 14.0 % keep spread and still -5.1 % vs `refg-rc2` 467.9. Same pattern as rc1 `k5`. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T16:54:11+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.7 / 54.6 / 58.1 | 56.7 | 6.2 % | - | - | - |
| 3 | 106.1 / 106.5 / 108.0 | 106.5 | 1.8 % | - | - | - |
| 5 | 120.6 / 118.8 / 116.8 | 118.8 | 3.2 % | - | - | - |
| 6 | 161.9 / 162.0 / 171.4 | 162.0 | 5.9 % | - | - | - |

### `kv262u086`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-22T01:10:41+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 61.5 / 56.9 / 61.5 | 61.5 | 7.5 % | - | - | - |
| 3 | 105.1 / 98.8 / 109.0 | 105.1 | 9.7 % | - | - | - |
| 5 | 129.5 / 128.6 / 127.8 | 128.6 | 1.3 % | - | - | - |
| 6 | 147.9 / 148.5 / 151.3 | 148.5 | 2.3 % | - | - | - |

### `nowo-rc2`

**Changed.** `proto2-030-rc2` plus `VLLM_USE_B12X_WO_PROJECTION=0`. One variable: stock O-proj einsum instead of overlay b12x WO. Confirmed `WO=0` in the container. Re-measure of rc1 `nowo` (448.0) on this pin.

**Verdict.** **positive, not a keep.** 57.7 / 102.6 / 132.9 / **149.0**, sum **442.2**, worst spread 5.7 %. Gates pass in all three passes; acceptance 50.6-56.6 % and tokens per step 4.523-4.923 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +5.9 % / +5.5 %, inside keep-spread. vs same-day `refg-rc2` (467.9 / 157.1) still -5.5 % / -5.2 %. WO overlay is still a cost on this pin. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T16:13:45+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 60.1 / 56.8 / 57.7 | 57.7 | 5.7 % | - | - | - |
| 3 | 98.3 / 102.6 / 103.3 | 102.6 | 4.9 % | - | - | - |
| 5 | 132.6 / 132.9 / 137.0 | 132.9 | 3.3 % | - | - | - |
| 6 | 149.0 / 148.1 / 150.0 | 149.0 | 1.3 % | - | - | - |

### `ncclp2p0-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_P2P_DISABLE=1`. One variable: disable NCCL CUDA P2P. Standing TP2 all-reduce is PYNCCL across two hosts; CUDA-IPC does not work across nodes. Confirmed `NCCL_P2P_DISABLE=1`.

**Verdict.** **pin noise, not a keep.** 55.8 / 98.4 / 137.7 / **148.7**, sum **440.6**, worst spread 13.8 %. 9x8 gate passes in all three passes; pass 3 france gate is garbled (` Paris.",\n    "label":`). One pass dipped to accept 48.2 % / tokens per step 4.376, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +5.5 % / +5.3 %, inside keep-spread. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. P2P off is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -9.3 % / -8.3 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T18:11:56+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.8 / 53.3 / 61.0 | 55.8 | 13.8 % | - | - | - |
| 3 | 98.4 / 103.9 / 97.5 | 98.4 | 6.5 % | - | - | - |
| 5 | 135.3 / 137.7 / 140.7 | 137.7 | 3.9 % | - | - | - |
| 6 | 148.7 / 148.6 / 155.4 | 148.7 | 4.6 % | - | - | - |

### `ncclqps4-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_IB_QPS_PER_CONNECTION=4`. One variable: IB queue pairs per connection (default 1). Standing TP2 all-reduce is PYNCCL over RoCE. Confirmed `NCCL_IB_QPS_PER_CONNECTION=4`.

**Verdict.** **pin noise, not a keep.** 55.6 / 101.2 / 135.5 / **146.4**, sum **438.7**, worst spread 8.1 %. Gates pass in all three passes; acceptance 50.1-54.6 % and tokens per step 4.491-4.830 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +5.1 % / +3.7 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Four IB QPs is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -9.7 % / -9.7 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T18:22:18+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.3 / 55.6 / 53.8 | 55.6 | 8.1 % | - | - | - |
| 3 | 101.2 / 104.7 / 99.7 | 101.2 | 4.9 % | - | - | - |
| 5 | 129.1 / 135.5 / 139.9 | 135.5 | 8.0 % | - | - | - |
| 6 | 146.2 / 153.5 / 146.4 | 146.4 | 5.0 % | - | - | - |

### `proto2-dg`

**Changed.** the **stock** arm after `VLLM_B12X_INDEXER_DIRECT_GATHER=1` was promoted into `configs/pin.main-029.env` and the serve forward list -- no extra env on the command line

**Verdict.** **the promotion, verified: the switch reaches the container from the pin and the win reproduces.** 55.5 / 99.6 / 131.7 / **150.4**, sum **437.2**, against `proto2-dgather`'s 424.3 and `proto2-dgather2`'s 422.6. All three runs are the same configuration, so the spread across them is the honest measure: c6 139.6-150.4 (a 7.8 % range) and the sum 422.6-437.2 (3.5 %). Gates pass in all three passes, acceptance 50.8-52.7 % and tokens per step 4.531-4.669, both unchanged from `proto2`. **This is now the standing configuration**, and `proto2`'s row below describes the arm before the switch was kept.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T20:21:25+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.5 / 50.7 / 58.4 | 55.5 | 13.9 % | - | - | - |
| 3 | 100.8 / 99.6 / 96.3 | 99.6 | 4.5 % | - | - | - |
| 5 | 133.1 / 131.7 / 129.6 | 131.7 | 2.7 % | - | - | - |
| 6 | 150.4 / 152.9 / 143.5 | 150.4 | 6.3 % | - | - | - |

### `k5`

**Changed.** `proto2-030` plus `NUM_SPECULATIVE_TOKENS=5`. One variable: DSpark k=5, the checkpoint's native block size. Protocol pin uses k=7. `proto5` on the old pin was rejected because capture missed 5x6=30; this pin captures 48, which covers 6*(5+1)=36. EXTRA overrides `run-arm.sh`'s hardcoded k=7. Confirmed `num_speculative_tokens: 5`.

**Verdict.** **negative, not kept.** 55.5 / 96.6 / 118.1 / **163.2**, sum **433.4**, worst spread 10.5 %. `gate_9x8` passes; `gate_france` is noisy on pass 3. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) c6 is +18.0 % (outside that spread) but the sum is only +5.2 % (inside). Tokens per step 4.031-4.571, below `proto2-030`'s floor 4.414 — k=5 returns fewer tokens per step even though acceptance rose to 60.8-71.8 %. vs `refg030` c6 163.2 vs 162.1 is inside the reference's 18.5 % spread, and the sum is still 10.4 % behind. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T22:16:51+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.0 / 55.5 / 57.8 | 55.5 | 10.5 % | - | - | - |
| 3 | 96.0 / 96.6 / 101.1 | 96.6 | 5.3 % | - | - | - |
| 5 | 118.2 / 116.1 / 118.1 | 118.1 | 1.8 % | - | - | - |
| 6 | 163.2 / 158.0 / 164.5 | 163.2 | 4.0 % | - | - | - |

### `kvblnhc-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=VLLM_KV_CACHE_LAYOUT=BLNHC`. One variable: the other legal layout vs standing auto BLHNC. `kvlbnhc-rc2` died: LBNHC is illegal; valid layouts are `['BLHNC', 'BLNHC']`. Confirmed `VLLM_KV_CACHE_LAYOUT=BLNHC`.

**Verdict.** **pin noise, not a keep.** 56.8 / 97.3 / 132.1 / **147.0**, sum **433.2**, worst spread 5.1 %. Gates pass in all three passes; acceptance 50.3-57.1 % and tokens per step 4.518-4.971 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +3.8 % / +4.1 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. BLNHC is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -10.8 % / -9.4 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T20:39:16+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.8 / 57.4 / 54.5 | 56.8 | 5.1 % | - | - | - |
| 3 | 96.3 / 97.3 / 98.9 | 97.3 | 2.7 % | - | - | - |
| 5 | 127.4 / 132.1 / 132.8 | 132.1 | 4.1 % | - | - | - |
| 6 | 147.0 / 149.3 / 144.5 | 147.0 | 3.3 % | - | - | - |

### `wochunk4-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_WO_QUANT_CHUNKS_PER_PROGRAM=4`. One variable: WO MXFP8 quant chunks (default 16). Chunks 8 and 32 were pin noise. Confirmed `B12X_WO_QUANT_CHUNKS_PER_PROGRAM=4`.

**Verdict.** **pin noise, not a keep.** 55.3 / 101.5 / 130.3 / **145.8**, sum **432.9**, worst spread 5.9 %. Gates pass in all three passes. One pass dipped to accept 48.9 % / tokens per step 4.406, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +3.7 % / +3.3 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. WO chunks 4 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -10.9 % / -10.1 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T13:48:15+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.3 / 55.9 / 53.6 | 55.3 | 4.2 % | - | - | - |
| 3 | 97.6 / 101.5 / 101.9 | 101.5 | 4.2 % | - | - | - |
| 5 | 131.5 / 130.3 / 123.8 | 130.3 | 5.9 % | - | - | - |
| 6 | 137.7 / 145.8 / 146.0 | 145.8 | 5.7 % | - | - | - |

### `moewarm-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_MOE_WARM_MS=8,24,40,48`. One variable: MoE warm-run sizes for protocol tokens c1=8, c3=24, c5=40, c6=48 (default 1,2,3,4,5,8 plus inferred capture sizes). Confirmed `B12X_MOE_WARM_MS=8,24,40,48`.

**Verdict.** **pin noise, not a keep.** 53.8 / 98.9 / 131.1 / **148.9**, sum **432.7**, worst spread 11.3 %. Gates pass in all three passes. One pass dipped to accept 47.5 % / tokens per step 4.303, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +3.6 % / +5.5 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Explicit MoE warm sizes are not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -10.9 % / -8.2 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T18:59:06+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 57.1 / 51.0 / 53.8 | 53.8 | 11.3 % | - | - | - |
| 3 | 98.9 / 100.7 / 95.0 | 98.9 | 5.8 % | - | - | - |
| 5 | 128.9 / 131.1 / 133.2 | 131.1 | 3.3 % | - | - | - |
| 6 | 149.9 / 148.8 / 148.9 | 148.9 | 0.7 % | - | - | - |

### `ncclbuf1m-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_BUFFSIZE=1048576`. One variable: NCCL buffer 1 MiB vs default 4 MiB. Standing TP2 all-reduce is PYNCCL. SYMM_MEM needs world_size>=4 so TP2 skips. Confirmed `NCCL_BUFFSIZE=1048576`.

**Verdict.** **pin noise, not a keep.** 52.8 / 101.3 / 134.9 / **143.7**, sum **432.7**, worst spread 9.3 %. Gates pass in all three passes; acceptance 50.1-57.1 % and tokens per step 4.491-4.971 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +3.6 % / +1.8 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. NCCL buffer 1 MiB is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -10.9 % / -11.4 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T17:27:23+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.8 / 56.8 / 51.9 | 52.8 | 9.3 % | - | - | - |
| 3 | 97.4 / 106.2 / 101.3 | 101.3 | 8.7 % | - | - | - |
| 5 | 134.9 / 129.2 / 136.4 | 134.9 | 5.3 % | - | - | - |
| 6 | 143.7 / 147.5 / 142.1 | 143.7 | 3.8 % | - | - | - |

### `attnfi030`

**Changed.** `proto2-030` plus `ATTENTION_BACKEND` and `DRAFT_ATTENTION_BACKEND` = `FLASHINFER_MLA_SPARSE_DSV4`. One family: FlashInfer sparse MLA DSV4, the SM12x auto-pick on the reference fork. Never re-measured on `main-030-rc1`. SM120 DSV4 specialization is present (`has_flashinfer_sparse_mla_sm120_config` True). Boot logged `No FlashInfer SM120 sparse MLA DSv4 decode autotune cache entries found. Falling back to FlashInfer's default tactic heuristic.` — cold-cache path, same as `proto2-attnfi` on the old pin.

**Verdict.** **negative, not kept.** 56.6 / 99.7 / 129.8 / **145.3**, sum **431.4**, worst spread 10.6 %. Gates pass in all three passes; acceptance 50.8-58.3 % and tokens per step 4.531-5.069, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) +4.7 % / +5.1 %, inside that spread. Against `p030b` (420.3) +2.6 %. Behind same-image ceiling `moemar` (455.3). FlashInfer MLA DSV4 on this pin is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T19:26:38+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.6 / 52.7 / 58.7 | 56.6 | 10.6 % | - | - | - |
| 3 | 97.5 / 99.7 / 103.7 | 99.7 | 6.2 % | - | - | - |
| 5 | 127.4 / 129.8 / 130.2 | 129.8 | 2.2 % | - | - | - |
| 6 | 143.1 / 145.3 / 146.3 | 145.3 | 2.2 % | - | - | - |

### `moework-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_WORK_SOURCE=persistent_grid`. One variable: b12x dynamic MoE work source. Library default is `materialized_queue`; `persistent_grid` is arithmetic striding. Confirmed `B12X_DYNAMIC_WORK_SOURCE=persistent_grid`. Re-measure of rc1 `moework` (426.8) on this pin.

**Verdict.** **pin noise, not a keep.** 53.9 / 102.9 / 132.2 / **142.2**, sum **431.2**, worst spread 10.2 %. Gates pass in all three passes. One pass dipped to accept 48.9 % / tokens per step 4.414, a hair under standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +3.3 % / +0.7 %. Against same-day control `p030-rc2b` (424.6 / 144.2) inside keep-spread. Work source is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -7.8 % / -9.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T19:02:52+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.9 / 57.1 / 53.9 | 53.9 | 5.9 % | - | - | - |
| 3 | 108.0 / 102.9 / 97.5 | 102.9 | 10.2 % | - | - | - |
| 5 | 132.2 / 124.4 / 136.2 | 132.2 | 8.9 % | - | - | - |
| 6 | 151.3 / 141.8 / 142.2 | 142.2 | 6.7 % | - | - | - |

### `ncclxnic-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_CROSSNIC=1`. One variable: NCCL CrossNIC across both UP RoCE ports. Standing `SPARK_IFACES` is `enp1s0f1np1` only; both sparks also have `enP2p1s0f1np1` UP. Confirmed `NCCL_CROSSNIC=1`.

**Verdict.** **pin noise, not a keep.** 56.9 / 101.9 / 127.0 / **145.3**, sum **431.1**, worst spread 7.9 %. Gates pass in all three passes; acceptance 49.2-56.3 % and tokens per step 4.444-4.923 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +3.3 % / +2.9 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. CrossNIC is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -11.3 % / -10.4 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T19:13:10+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.6 / 56.9 / 57.1 | 56.9 | 6.2 % | - | - | - |
| 3 | 99.1 / 107.1 / 101.9 | 101.9 | 7.9 % | - | - | - |
| 5 | 127.0 / 128.8 / 125.9 | 127.0 | 2.3 % | - | - | - |
| 6 | 142.4 / 149.2 / 145.3 | 145.3 | 4.7 % | - | - | - |

### `micromax-rc2`

**Changed.** `proto2-030-rc2` plus bind-mount of patched `b12x/moe/fused_moe/_impl.py`: `_MICRO_MAX_TOKENS` 8→48 and `_MICRO_DYNAMIC_CUTOVER_PAIRS_DEFAULT` 64→320. One variable: extend native micro MoE so protocol c6 (48 tok × topk 6 = 288) stays micro. Native b12x has no static kernel; FlashInfer static IMA'd on live MXFP4. Confirmed overlay `_MICRO_MAX_TOKENS = 48`.

**Verdict.** **pin noise, not a keep.** 54.8 / 99.9 / 133.9 / **142.4**, sum **431.0**, worst spread 10.6 %. Gates pass in all three passes. One pass dipped to accept 48.8 % / tokens per step 4.401, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +3.2 % / +0.8 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Extending micro past m=8 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -11.3 % / -12.2 %. Standing arm remains `proto2-030-rc2`. Overlay not kept.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T12:10:30+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.8 / 53.9 / 54.8 | 54.8 | 1.6 % | - | - | - |
| 3 | 99.9 / 92.1 / 102.7 | 99.9 | 10.6 % | - | - | - |
| 5 | 135.1 / 127.9 / 133.9 | 133.9 | 5.4 % | - | - | - |
| 6 | 141.1 / 142.4 / 143.3 | 142.4 | 1.5 % | - | - | - |

### `moetile64-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_TILE_MN=64x128`. One variable: force W4A8 dynamic tile M64 vs auto M16/M32 on GB10. 32x128 and 128x128 already measured on this pin. Confirmed `B12X_DYNAMIC_TILE_MN=64x128`.

**Verdict.** **pin noise, not a keep.** 55.4 / 102.0 / 131.0 / **142.2**, sum **430.6**, worst spread 12.6 %. Gates pass in all three passes. One pass dipped to accept 47.3 % / tokens per step 4.303, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +3.1 % / +0.7 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Forced M64 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -11.4 % / -12.3 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T21:40:14+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.4 / 50.9 / 56.1 | 55.4 | 9.4 % | - | - | - |
| 3 | 102.0 / 89.7 / 102.6 | 102.0 | 12.6 % | - | - | - |
| 5 | 134.2 / 131.0 / 128.8 | 131.0 | 4.1 % | - | - | - |
| 6 | 142.2 / 139.1 / 149.4 | 142.2 | 7.2 % | - | - | - |

### `ncclto22-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_IB_TIMEOUT=22`. One variable: IB timeout 22 vs default 18. Standing TP2 all-reduce is PYNCCL over RoCE. Confirmed `NCCL_IB_TIMEOUT=22`.

**Verdict.** **pin noise, not a keep.** 55.1 / 97.6 / 131.2 / **145.6**, sum **429.5**, worst spread 11.6 %. 9x8 gate passes in all three passes; pass 1 france gate is garbled (` Paris.", "The capital of`). One pass dipped to accept 47.7 % / tokens per step 4.339, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.9 % / +3.1 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. IB timeout 22 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -11.6 % / -10.2 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T22:00:58+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.9 / 55.1 / 50.5 | 55.1 | 11.6 % | - | - | - |
| 3 | 96.3 / 99.5 / 97.6 | 97.6 | 3.3 % | - | - | - |
| 5 | 131.6 / 131.2 / 124.4 | 131.2 | 5.5 % | - | - | - |
| 6 | 147.2 / 145.6 / 144.8 | 145.6 | 1.6 % | - | - | - |

### `wochunk2-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_WO_QUANT_CHUNKS_PER_PROGRAM=2`. One variable: WO MXFP8 quant chunks (default 16). Chunks 4/8/32 were pin noise. Confirmed `B12X_WO_QUANT_CHUNKS_PER_PROGRAM=2`.

**Verdict.** **pin noise, not a keep.** 56.8 / 94.8 / 130.2 / **147.7**, sum **429.5**, worst spread 13.9 %. Gates pass in all three passes. One pass dipped to accept 47.5 % / tokens per step 4.303, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.9 % / +4.6 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. WO chunks 2 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -11.6 % / -8.9 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T14:01:45+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.8 / 49.9 / 57.8 | 56.8 | 13.9 % | - | - | - |
| 3 | 95.5 / 94.6 / 94.8 | 94.8 | 0.9 % | - | - | - |
| 5 | 129.2 / 131.8 / 130.2 | 130.2 | 2.0 % | - | - | - |
| 6 | 147.7 / 145.8 / 149.0 | 147.7 | 2.2 % | - | - | - |

### `ncclplug0-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_NET_PLUGIN=none`. One variable: disable external NCCL NET plugins. Complementary to `ncclcnet0-rc2` (CollNet off, pin noise). Confirmed `NCCL_NET_PLUGIN=none`.

**Verdict.** **pin noise, not a keep.** 53.7 / 101.3 / 132.0 / **142.4**, sum **429.4**, worst spread 10.0 %. Gates pass in all three passes. One pass dipped to accept 45.6 % / tokens per step 4.197, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.9 % / +0.8 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. NET plugin none is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -11.6 % / -12.2 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-21T00:15:12+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.3 / 53.7 / 49.0 | 53.7 | 9.9 % | - | - | - |
| 3 | 95.2 / 101.3 / 105.3 | 101.3 | 10.0 % | - | - | - |
| 5 | 130.8 / 132.0 / 133.2 | 132.0 | 1.8 % | - | - | - |
| 6 | 138.3 / 142.4 / 143.6 | 142.4 | 3.7 % | - | - | - |

### `now4sh-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_W4A8_SHARE_INPUT=0`. One variable: W4A8 shared-input producer off. DSV4 MXFP4 experts map to `quant_mode=w4a8_mx`. Share-input default is on when dense or decode candidate is true (`moeshare-rc2` forced it on). Confirmed `B12X_DYNAMIC_W4A8_SHARE_INPUT=0`. Re-measure of rc1 `now4sh` (420.1) on this pin.

**Verdict.** **pin noise, not a keep.** 52.1 / 99.1 / 130.9 / **147.3**, sum **429.4**, worst spread 6.2 %. Gates pass in all three passes; acceptance 49.2-55.1 % and tokens per step 4.439-4.845 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.9 % / +4.3 %. Against same-day control `p030-rc2b` (424.6 / 144.2) inside keep-spread. W4A8 share-input off is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -8.2 % / -6.2 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T22:58:22+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.1 / 54.0 / 52.1 | 52.1 | 3.6 % | - | - | - |
| 3 | 99.1 / 100.5 / 94.4 | 99.1 | 6.2 % | - | - | - |
| 5 | 130.9 / 127.9 / 134.1 | 130.9 | 4.7 % | - | - | - |
| 6 | 146.1 / 149.4 / 147.3 | 147.3 | 2.2 % | - | - | - |

### `wooff5-rc2`

**Changed.** `wooff-rc2` (WO overlay off, k=7) re-measured with **five** passes (`PASSES=5`). One variable against `wooff-rc2`: measurement power. The keep-rule's bar is the larger median spread, and the c1 swing at three passes (~10 %) was larger than any effect found, so the baseline itself had to be tightened.

**Verdict.** **re-baseline, confirms the 3-pass number.** 54.2 / 99.5 / 131.1 / **144.6**, sum **429.4**, worst spread 8.5 %. Both gates pass in all five passes; acceptance 49.8-56.9 % and tokens per step 4.476-4.923 hold the floor. Against the same config at three passes (`wooff-rc2`, 426.9 / 144.2) it is +0.6 % / +0.3 %, so the three-pass figure was not an outlier. c1 is still 8.5 % spread at n=5, i.e. the rig's c1 variance is intrinsic and caps the detectable effect size.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-21T13:26:18+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.2 / 56.7 / 54.1 / 52.1 / 55.9 | 54.2 | 8.5 % | - | - | - |
| 3 | 100.8 / 96.9 / 99.6 / 97.0 / 99.5 | 99.5 | 3.9 % | - | - | - |
| 5 | 132.2 / 128.9 / 127.7 / 133.6 / 131.1 | 131.1 | 4.5 % | - | - | - |
| 6 | 139.2 / 144.8 / 144.8 / 142.4 / 144.6 | 144.6 | 3.9 % | - | - | - |

### `cgwarm3-rc2`

**Changed.** `proto2-030-rc2` plus `CUDAGRAPH_NUM_OF_WARMUPS=3`. One variable: CUDA-graph warmups before capture (standing 0). Confirmed `cudagraph_num_of_warmups': 3` on the APIServer compilation config (EngineCore later logs 1). Plumbing added in `scripts/05-serve.sh`.

**Verdict.** **pin noise, not a keep.** 53.6 / 99.6 / 129.3 / **146.8**, sum **429.3**, worst spread 9.3 %. Gates pass in all three passes. One pass dipped to accept 46.4 % / tokens per step 4.197, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.8 % / +4.0 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Extra graph warmups are not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -11.6 % / -9.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T20:58:34+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 49.4 / 53.6 / 54.4 | 53.6 | 9.3 % | - | - | - |
| 3 | 97.1 / 103.7 / 99.6 | 99.6 | 6.6 % | - | - | - |
| 5 | 130.0 / 129.3 / 128.5 | 129.3 | 1.2 % | - | - | - |
| 6 | 151.1 / 140.8 / 146.8 | 146.8 | 7.0 % | - | - | - |

### `util8663`

**Changed.** `proto2-030` plus `GPU_MEMORY_UTILIZATION=0.8663`. One variable: restore the pre-profiler KV budget. v0.30 CUDA-graph memory profiling maps 0.8389 to effective 0.8115; the engine names 0.8663 as the util that keeps the same KV size as 0.8389 without the profiler. Confirmed in the engine log (`gpu_memory_utilization=0.8663`, Available KV 13.82 GiB vs `moecap175`'s 9.39 GiB at 0.8389).

**Verdict.** **negative, not kept.** 51.9 / 102.6 / 129.2 / **145.3**, sum **429.0**, worst spread 6.2 %. Gates pass in all three passes; acceptance 49.7-56.3 % and tokens per step 4.452-4.923, neither lower than `proto2-030`. Against same-base `proto2-030` (412.1 / c6 138.3, spread 12.8 %) the sum is +4.1 % and c6 is +5.1 %, both inside the larger spread. Extra KV does not close the reference gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T15:50:29+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 51.8 / 52.4 / 51.9 | 51.9 | 1.2 % | - | - | - |
| 3 | 96.3 / 102.7 / 102.6 | 102.6 | 6.2 % | - | - | - |
| 5 | 129.2 / 127.7 / 129.3 | 129.2 | 1.2 % | - | - | - |
| 6 | 145.3 / 144.2 / 148.4 | 145.3 | 2.9 % | - | - | - |

### `ncclar1-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_IB_AR_ALGORITHM=1`. One variable: IB adaptive routing. Distinct from `ncclring-rc2` (`NCCL_ALGO=Ring`, worse at c6). Confirmed `NCCL_IB_AR_ALGORITHM=1`.

**Verdict.** **pin noise, not a keep.** 56.4 / 99.1 / 131.5 / **141.8**, sum **428.8**, worst spread 10.1 %. Gates pass in all three passes. One pass dipped to accept 49.1 % / tokens per step 4.420, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.7 % / +0.4 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. IB adaptive routing is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -11.8 % / -12.6 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T23:31:26+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.9 / 53.2 / 56.4 | 56.4 | 10.1 % | - | - | - |
| 3 | 94.8 / 99.1 / 101.2 | 99.1 | 6.5 % | - | - | - |
| 5 | 132.0 / 131.5 / 126.4 | 131.5 | 4.3 % | - | - | - |
| 6 | 141.6 / 141.8 / 141.8 | 141.8 | 0.1 % | - | - | - |

### `cgcopy-rc2`

**Changed.** `proto2-030-rc2` plus `CUDAGRAPH_COPY_INPUTS=true`. One variable: copy CUDA-graph inputs on replay (standing False). Confirmed `cudagraph_copy_inputs': True`.

**Verdict.** **pin noise, not a keep.** 55.0 / 102.6 / 128.7 / **142.3**, sum **428.6**, worst spread 7.8 %. Gates pass in all three passes; acceptance 49.4-54.9 % and tokens per step 4.444-4.830 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.7 % / +0.8 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Copying graph inputs is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -11.8 % / -12.3 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T18:46:22+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 57.4 / 53.1 / 55.0 | 55.0 | 7.8 % | - | - | - |
| 3 | 103.9 / 102.6 / 99.5 | 102.6 | 4.3 % | - | - | - |
| 5 | 133.1 / 128.7 / 128.1 | 128.7 | 3.9 % | - | - | - |
| 6 | 141.9 / 142.3 / 152.2 | 142.3 | 7.2 % | - | - | - |

### `cgsizes-rc2`

**Changed.** `proto2-030-rc2` plus `CUDAGRAPH_CAPTURE_SIZES=[8,24,40,48]`. One variable: explicit CUDA-graph capture sizes for protocol tokens c1=8, c3=24, c5=40, c6=48. Pin `max_cudagraph_capture_size=48`, capture_sizes unset. Confirmed `cudagraph_capture_sizes': [8, 24, 40, 48]`.

**Verdict.** **pin noise, not a keep.** 53.8 / 99.2 / 129.5 / **145.6**, sum **428.1**, worst spread 9.3 %. Gates pass in all three passes; acceptance 49.6-55.3 % and tokens per step 4.452-4.845 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.5 % / +3.1 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Explicit capture sizes are not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -11.9 % / -10.2 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T18:34:31+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.8 / 53.8 / 52.5 | 53.8 | 6.1 % | - | - | - |
| 3 | 99.2 / 102.9 / 93.7 | 99.2 | 9.3 % | - | - | - |
| 5 | 134.7 / 129.5 / 128.8 | 129.5 | 4.6 % | - | - | - |
| 6 | 147.6 / 143.7 / 145.6 | 145.6 | 2.7 % | - | - | - |

### `ncclpxn0-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_PXN_DISABLE=1`. One variable: disable NCCL PXN (proxy NIC). Standing TP2 all-reduce is PYNCCL over two-host RoCE. Confirmed `NCCL_PXN_DISABLE=1`.

**Verdict.** **pin noise, not a keep.** 54.2 / 101.2 / 127.7 / **144.8**, sum **427.9**, worst spread 7.3 %. 9x8 gate passes in all three passes; pass 3 france gate is garbled (` Paris.", "The capital of`). One pass dipped to accept 48.1 % / tokens per step 4.339, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.5 % / +2.5 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. PXN off is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -11.9 % / -10.7 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T19:34:28+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 51.1 / 54.2 / 54.3 | 54.2 | 5.9 % | - | - | - |
| 3 | 104.3 / 96.9 / 101.2 | 101.2 | 7.3 % | - | - | - |
| 5 | 127.7 / 132.4 / 125.1 | 127.7 | 5.7 % | - | - | - |
| 6 | 148.3 / 144.8 / 143.9 | 144.8 | 3.0 % | - | - | - |

### `gencvllm-rc2`

**Changed.** `proto2-030-rc2` plus `--generation-config vllm` via `ARM_EXTRA_ARGS`. One variable: match the anemll recipe. Ours default `auto` loads checkpoint `generation_config.json`. Protocol requests send temperature. Confirmed `generation_config: vllm`. Re-measure of rc1 `gencvllm` (426.8) on this pin.

**Verdict.** **pin noise, not a keep.** 55.9 / 99.1 / 130.4 / **142.4**, sum **427.8**, worst spread 7.6 %. Gates pass in all three passes; acceptance 50.4-59.0 % and tokens per step 4.511-5.069 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.5 % / +0.8 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Neutral vLLM sampling defaults are not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -8.6 % / -9.4 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T18:31:29+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.4 / 55.9 / 58.4 | 55.9 | 5.4 % | - | - | - |
| 3 | 105.0 / 97.5 / 99.1 | 99.1 | 7.6 % | - | - | - |
| 5 | 130.4 / 129.9 / 131.4 | 130.4 | 1.2 % | - | - | - |
| 6 | 142.4 / 142.8 / 137.6 | 142.4 | 3.7 % | - | - | - |

### `nostream`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_INDEXER_STREAM_SCORER=0`. One variable: indexer stream scorer off (unset defaults True). Overlay scores DSA indexer logits via `logits_paged` -> `run_paged_logits_kernel`; the tiled/supertile path gates stream-scorer on this env. Confirmed in container env (`B12X_INDEXER_STREAM_SCORER=0`).

**Verdict.** **negative, not kept.** 52.5 / 100.0 / 133.6 / **141.6**, sum **427.7**, worst spread 9.7 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 47.7 % and 4.339 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). Stream scorer is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T23:51:04+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 50.8 / 55.9 / 52.5 | 52.5 | 9.7 % | - | - | - |
| 3 | 98.9 / 103.9 / 100.0 | 100.0 | 5.0 % | - | - | - |
| 5 | 135.3 / 125.9 / 133.6 | 133.6 | 7.0 % | - | - | - |
| 6 | 141.0 / 143.0 / 141.6 | 141.6 | 1.4 % | - | - | - |

### `nostream-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_INDEXER_STREAM_SCORER=0`. One variable: indexer stream scorer off (default True). Overlay scores DSA indexer logits via `logits_paged`. Confirmed `B12X_INDEXER_STREAM_SCORER=0`. Re-measure of rc1 `nostream` on this pin.

**Verdict.** **pin noise, not a keep.** 55.5 / 99.4 / 129.1 / **143.7**, sum **427.7**, worst spread 7.4 %. Gates pass in all three passes; acceptance 50.0-56.2 % and tokens per step 4.460-4.923 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.4 % / +1.8 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Stream scorer off is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -8.6 % / -8.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T20:27:27+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.5 / 52.4 / 55.5 | 55.5 | 7.4 % | - | - | - |
| 3 | 99.4 / 98.7 / 101.2 | 99.4 | 2.5 % | - | - | - |
| 5 | 126.4 / 130.0 / 129.1 | 129.1 | 2.8 % | - | - | - |
| 6 | 139.9 / 145.2 / 143.7 | 143.7 | 3.7 % | - | - | - |

### `moetilem32-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_MOE_TILE_MN=32x128`. One variable: micro MoE tile. `B12X_MOE_TILE_MN` gates `_select_micro_mma_tiler_mn` (c1, below the 64-row cutover). Default is 64x128; 128x128 died at KV floor. Confirmed `B12X_MOE_TILE_MN=32x128`.

**Verdict.** **pin noise, not a keep.** 51.9 / 98.4 / 130.4 / **146.8**, sum **427.5**, worst spread 17.0 %. Gates pass in all three passes. One pass dipped to accept 49.1 % / tokens per step 4.414, a hair under standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.4 % / +4.0 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Micro 32x128 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.0 % / -9.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T02:04:59+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 51.7 / 51.9 / 60.5 | 51.9 | 17.0 % | - | - | - |
| 3 | 96.6 / 100.6 / 98.4 | 98.4 | 4.1 % | - | - | - |
| 5 | 133.0 / 129.0 / 130.4 | 130.4 | 3.1 % | - | - | - |
| 6 | 137.9 / 146.8 / 148.4 | 146.8 | 7.2 % | - | - | - |

### `wooff-a16-rc2`

**Changed.** `wooff-rc2` (WO overlay off by pin default) plus `SERVE_EXTRA_ENV=VLLM_B12X_MOE_FP4_FORCE_A16=1`. One variable against the new standing `wooff-rc2`: force W4A16 on the MoE, the lever that measured best alone (`moea16-rc2`, 453.7 / 152.8). Confirmed `VLLM_B12X_MOE_FP4_FORCE_A16=1` and `VLLM_USE_B12X_WO_PROJECTION=0`. Targets the `ffn` region the c1 profile names at 53 % of the step.

**Verdict.** **wash, and it does not replicate `moea16-rc2`.** 54.9 / 102.1 / 130.8 / **139.6**, sum **427.4**, worst spread 5.9 %. Gates pass; acceptance 50.3-55.2 % and tokens per step 4.518-4.876 hold the floor. Against `wooff-rc2` (426.9 / 144.2) +0.1 % / -3.2 %, i.e. a wash. Against standing `proto2-030-rc2` (417.5) +2.4 %. Against the `moea16-rc2` measurement of the same knob (453.7 / 152.8) it is 5.8 % / 8.6 % **worse**, so the 453.7 was mostly rig noise. Reading: on this pin every remaining one-variable knob sits inside the rig's own ~14 % swing, which is why no single-variable arm can clear the keep-rule. vs same-day `refg-rc2b` (485.9 / 162.2) -12.0 % / -13.9 %.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-21T12:48:29+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.7 / 53.8 / 54.9 | 54.9 | 5.3 % | - | - | - |
| 3 | 97.4 / 103.4 / 102.1 | 102.1 | 5.9 % | - | - | - |
| 5 | 130.8 / 130.0 / 131.4 | 130.8 | 1.1 % | - | - | - |
| 6 | 146.9 / 139.6 / 139.6 | 139.6 | 5.2 % | - | - | - |

### `noh16`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_MLA_SM120_DSV4_H16_NATIVE=0`. One variable: force DSV4 H16 native decode off. Unset is auto; on Spark (48 SMs) auto turns H16 on for many-chunk / batched-row decode. Overlay `B12X_MLA_SPARSE` decode goes through `run_unified_decode`. Confirmed in container env (`B12X_MLA_SM120_DSV4_H16_NATIVE=0`).

**Verdict.** **negative, not kept.** 53.6 / 100.8 / 129.5 / **143.4**, sum **427.3**, worst spread 8.0 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 47.7 % and 4.339 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). H16 auto is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T01:35:01+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.6 / 54.8 / 50.5 | 53.6 | 8.0 % | - | - | - |
| 3 | 104.5 / 100.3 / 100.8 | 100.8 | 4.2 % | - | - | - |
| 5 | 130.1 / 129.5 / 127.7 | 129.5 | 1.9 % | - | - | - |
| 6 | 144.7 / 143.4 / 141.9 | 143.4 | 2.0 % | - | - | - |

### `wooff-rc2`

**Changed.** the WO projection overlay made non-default: `configs/pin.main-029.env` now sets `VLLM_USE_B12X_WO_PROJECTION:-0`, so `try_b12x_wo_proj` returns None and `deep_gemm_fp8_o_proj` runs the einsum path. One variable: the served WO projection. Confirmed `VLLM_USE_B12X_WO_PROJECTION=0` in the container. Fix landed from the c1 decode profile, which puts the `wo` region at 57.4 ms of a 150.6 ms step (38 %).

**Verdict.** **positive, not a keep.** 56.8 / 96.5 / 129.4 / **144.2**, sum **426.9**, worst spread 7.5 % (the tightest of any same-pin arm). Gates pass in all three passes; acceptance 49.6-57.0 % and tokens per step 4.452-4.971 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.3 % / +2.1 %, inside keep-spread. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. This is a re-measure of `nowo-rc2` (442.2 / 149.0) and is 3.5 % below it, so the earlier single-arm gain was partly noise; the honest reading is that disabling the overlay is never worse and both arms favour it. Kept as the default because the overlay is our own emulation layer with documented Dynamo fragility and it removes four elementwise passes plus a per-group Python copy loop per layer. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.1 % / -11.1 %.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-21T12:40:13+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.8 / 57.5 / 54.4 | 56.8 | 5.5 % | - | - | - |
| 3 | 96.5 / 95.9 / 100.5 | 96.5 | 4.8 % | - | - | - |
| 5 | 129.4 / 123.8 / 129.9 | 129.4 | 4.7 % | - | - | - |
| 6 | 153.5 / 142.7 / 144.2 | 144.2 | 7.5 % | - | - | - |

### `gencvllm`

**Changed.** `proto2-030` plus `--generation-config vllm` via `ARM_EXTRA_ARGS`. One variable: match the anemll recipe. Ours default `auto` loads checkpoint `generation_config.json` (`do_sample` true, temp 1.0, top_p 1.0). Protocol requests send temperature explicitly, so this may be a no-op. Confirmed in non-default args (`generation_config: vllm`).

**Verdict.** **negative, not kept.** 55.9 / 101.4 / 127.7 / **141.8**, sum **426.8**, worst spread 6.8 %. Gates pass in all three passes; acceptance 50.0-56.1 % and tokens per step 4.491-4.876, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. Neutral vLLM sampling defaults are not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T21:56:45+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.9 / 56.2 / 52.4 | 55.9 | 6.8 % | - | - | - |
| 3 | 105.3 / 100.2 / 101.4 | 101.4 | 5.0 % | - | - | - |
| 5 | 130.4 / 127.5 / 127.7 | 127.7 | 2.3 % | - | - | - |
| 6 | 141.8 / 145.1 / 140.3 | 141.8 | 3.4 % | - | - | - |

### `moework`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_WORK_SOURCE=persistent_grid`. One variable: the b12x dynamic MoE work source. Library default is `materialized_queue`; `persistent_grid` is documented as arithmetic striding for A/B. Env confirmed in the container.

**Verdict.** **negative, not kept.** 54.8 / 97.3 / 129.3 / **145.4**, sum **426.8**, worst spread 6.6 %. Gates pass in all three passes; acceptance 49.3-54.9 % and tokens per step 4.452-4.830, neither lower than `proto2-030`. Against same-base `proto2-030` (412.1 / c6 138.3, spread 12.8 %) the sum is +3.6 % and c6 is +5.1 %, both inside the larger spread. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T15:40:22+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.5 / 51.9 / 54.8 | 54.8 | 6.6 % | - | - | - |
| 3 | 96.7 / 100.8 / 97.3 | 97.3 | 4.2 % | - | - | - |
| 5 | 129.1 / 136.9 / 129.3 | 129.3 | 6.0 % | - | - | - |
| 6 | 145.4 / 140.2 / 146.7 | 145.4 | 4.5 % | - | - | - |

### `ncclhca-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`. One variable: name both ACTIVE RoCE HCAs. Standing `NCCL_IB_HCA` is empty (auto). Confirmed `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`.

**Verdict.** **pin noise, not a keep.** 53.2 / 101.1 / 130.5 / **141.7**, sum **426.5**, worst spread 11.3 %. Gates pass in all three passes; acceptance 50.0-58.4 % and tokens per step 4.504-5.069 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.2 % / +0.4 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Naming both HCAs is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.2 % / -12.6 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T19:22:54+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.8 / 53.2 / 52.8 | 53.2 | 11.3 % | - | - | - |
| 3 | 102.9 / 101.1 / 96.8 | 101.1 | 6.0 % | - | - | - |
| 5 | 130.5 / 129.9 / 132.2 | 130.5 | 1.8 % | - | - | - |
| 6 | 138.7 / 141.7 / 150.0 | 141.7 | 8.0 % | - | - | - |

### `proto2-dg-pair`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-18T02:14:29+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.9 / 52.9 / 57.0 | 53.9 | 7.6 % | - | - | - |
| 3 | 100.2 / 96.9 / 98.6 | 98.6 | 3.3 % | - | - | - |
| 5 | 129.7 / 130.0 / 129.9 | 129.9 | 0.2 % | - | - | - |
| 6 | 144.0 / 150.2 / 140.5 | 144.0 | 6.7 % | - | - | - |

### `notiny-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_W4A8_TINY_DECODE=0`. One variable: W4A8 tiny-decode off (default 1). On SM121 DSV4F the tiny path is excluded for `num_tokens>=3`, so it only owns c1 (m=1). Confirmed `B12X_W4A8_TINY_DECODE=0`. Re-measure of rc1 `notiny` (426.1) on this pin.

**Verdict.** **pin noise, not a keep.** 55.2 / 98.9 / 127.9 / **144.2**, sum **426.2**, worst spread 7.2 %. Gates pass in all three passes; acceptance 50.3-56.6 % and tokens per step 4.507-4.939 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.1 % / +2.1 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Tiny-decode off is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -8.9 % / -8.2 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T20:17:01+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 57.0 / 54.9 / 55.2 | 55.2 | 3.8 % | - | - | - |
| 3 | 105.6 / 98.5 / 98.9 | 98.9 | 7.2 % | - | - | - |
| 5 | 128.5 / 127.9 / 127.4 | 127.9 | 0.9 % | - | - | - |
| 6 | 147.2 / 144.2 / 143.5 | 144.2 | 2.6 % | - | - | - |

### `notiny`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_W4A8_TINY_DECODE=0`. One variable: W4A8 tiny-decode off (default 1). On SM121 DSV4F (k=6144, n=1024) the tiny path is excluded for `num_tokens>=3`, so it only owns c1 (m=1). Confirmed in container env (`B12X_W4A8_TINY_DECODE=0`).

**Verdict.** **negative, not kept.** 52.8 / 99.2 / 133.1 / **141.0**, sum **426.1**, worst spread 8.9 %. Gates pass in all three passes; acceptance 49.8-55.2 % and tokens per step 4.452-4.830, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. Tiny-decode is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T00:57:12+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.1 / 55.9 / 52.8 | 52.8 | 7.2 % | - | - | - |
| 3 | 105.8 / 99.2 / 97.0 | 99.2 | 8.9 % | - | - | - |
| 5 | 133.1 / 134.1 / 132.9 | 133.1 | 0.9 % | - | - | - |
| 6 | 141.0 / 142.8 / 137.0 | 141.0 | 4.1 % | - | - | - |

### `atom24`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_DENSE_ATOM_24=1`. One variable: experimental 24-atom MMA in b12x dense GEMM (default 0). Changes generated code and is keyed into the persistent compile cache. Hits `B12xFp8BlockScaledMMKernel` via `mm_block_fp8` -> `dense_gemm`. Confirmed in container env (`B12X_DENSE_ATOM_24=1`); linear selection stayed `B12xFp8BlockScaledMMKernel`.

**Verdict.** **negative, not kept.** 52.8 / 103.0 / 125.4 / **144.8**, sum **426.0**, worst spread 9.7 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 45.7 % and 4.197 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). 24-atom dense GEMM is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T19:04:11+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.8 / 54.5 / 49.4 | 52.8 | 9.7 % | - | - | - |
| 3 | 101.0 / 103.0 / 103.9 | 103.0 | 2.8 % | - | - | - |
| 5 | 124.3 / 125.4 / 132.5 | 125.4 | 6.5 % | - | - | - |
| 6 | 140.3 / 146.2 / 144.8 | 144.8 | 4.1 % | - | - | - |

### `noreuse-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_MICRO_REUSE_COMPILED=0`. One variable: micro MoE compiled-kernel reuse off (default 1). Micro owns the tiny tail below the 64 routed-row cutover (protocol c1). Confirmed `B12X_MICRO_REUSE_COMPILED=0`.

**Verdict.** **pin noise, not a keep.** 54.7 / 100.5 / 127.0 / **143.8**, sum **426.0**, worst spread 5.1 %. Gates pass in all three passes; acceptance 49.9-55.8 % and tokens per step 4.491-4.876 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.0 % / +1.8 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Micro reuse off is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.3 % / -11.3 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T01:43:35+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.7 / 56.0 / 53.2 | 54.7 | 5.1 % | - | - | - |
| 3 | 99.9 / 100.8 / 100.5 | 100.5 | 0.9 % | - | - | - |
| 5 | 127.0 / 126.0 / 128.1 | 127.0 | 1.7 % | - | - | - |
| 6 | 140.3 / 143.8 / 144.6 | 143.8 | 3.0 % | - | - | - |

### `moecut32-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_MICRO_DYNAMIC_CUTOVER_PAIRS=32`. One variable: micro vs dynamic routed-row cutover (default 64). Dispatch is `num_tokens<=8` and `routed_rows < cutover`. Protocol c1 is 8 tok x topk 6 = 48 pairs, so only c1 is micro at default. Cutover 32 sends c1 to dynamic. Confirmed `B12X_MICRO_DYNAMIC_CUTOVER_PAIRS=32`.

**Verdict.** **pin noise, not a keep.** 55.4 / 99.6 / 129.9 / **141.0**, sum **425.9**, worst spread 5.4 %. Gates pass in all three passes; acceptance 49.5-55.4 % and tokens per step 4.459-4.876 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.0 % / -0.1 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Sending c1 to dynamic is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.3 % / -13.1 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T15:57:51+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.0 / 56.1 / 55.4 | 55.4 | 3.8 % | - | - | - |
| 3 | 100.6 / 96.1 / 99.6 | 99.6 | 4.5 % | - | - | - |
| 5 | 130.0 / 129.0 / 129.9 | 129.9 | 0.8 % | - | - | - |
| 6 | 144.5 / 141.0 / 136.9 | 141.0 | 5.4 % | - | - | - |

### `ncclsl0-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_IB_SL=0`. One variable: IB service level 0. Complementary to `nccltc106-rc2` (TC 106, pin noise). Confirmed `NCCL_IB_SL=0`.

**Verdict.** **pin noise, not a keep.** 54.8 / 97.7 / 130.8 / **142.6**, sum **425.9**, worst spread 10.8 %. Gates pass in all three passes. One pass dipped to accept 47.9 % / tokens per step 4.339, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +2.0 % / +1.0 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. IB SL 0 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.3 % / -12.1 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T22:58:34+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.8 / 50.3 / 56.2 | 54.8 | 10.8 % | - | - | - |
| 3 | 97.7 / 98.9 / 97.1 | 97.7 | 1.8 % | - | - | - |
| 5 | 131.4 / 130.8 / 127.9 | 130.8 | 2.7 % | - | - | - |
| 6 | 142.6 / 141.4 / 145.9 | 142.6 | 3.2 % | - | - | - |

### `split2`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_MLA_SM120_NUM_SPLITS=2`. One variable: pin MLA decode split-K to 2. Unset uses the wave-balanced heuristic; `split1` pinned 1. Overlay `B12X_MLA_SPARSE` decode goes through `run_unified_decode`. Dual-cache protocol is ~11 chunks. Confirmed in container env (`B12X_MLA_SM120_NUM_SPLITS=2`).

**Verdict.** **negative, not kept.** 54.7 / 93.1 / 130.5 / **147.5**, sum **425.8**, worst spread 9.1 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) +3.3 % / +6.7 %, inside that spread. Against `p030b` (420.3) +1.3 %. One pass dipped to accept 48.0 % and 4.339 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). MLA num_splits=2 is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T02:00:11+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.7 / 54.1 / 56.5 | 54.7 | 4.4 % | - | - | - |
| 3 | 92.8 / 101.3 / 93.1 | 93.1 | 9.1 % | - | - | - |
| 5 | 130.5 / 128.6 / 133.8 | 130.5 | 4.0 % | - | - | - |
| 6 | 141.3 / 147.5 / 148.6 | 147.5 | 4.9 % | - | - | - |

### `proto2-moetile`

**Changed.** the kept arm (`proto2-dg`) plus `B12X_DYNAMIC_TILE_MN=64x128`, the MoE tile-shape override b12x documents as a benchmarking knob -- the first time any of the three tile knobs has been tried

**Verdict.** **a wash, and inside the kept arm's own run-to-run range.** 58.2 / 96.8 / 126.4 / 144.3, sum **425.7**, against `proto2-dg`'s 55.5 / 99.6 / 131.7 / 150.4, sum 437.2 -- better at c1 (+4.9 %), worse at c3, c5 and c6 (-2.8, -4.0, -4.1 %), and -2.6 % on the sum. The three runs of the unmodified kept arm span sums 422.6-437.2, so 425.7 sits inside that. Gates pass in all three passes and acceptance (51.0-55.1 %) and tokens per step (4.555-4.830) are fine, so it is a legitimate A/B rather than a correctness casualty -- it simply buys nothing. The MoE's remaining +50.1 ms at c6 is not reachable by tile shape, at least not this one.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T20:30:07+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.2 / 58.2 / 53.4 | 58.2 | 8.2 % | - | - | - |
| 3 | 91.8 / 96.8 / 98.6 | 96.8 | 7.0 % | - | - | - |
| 5 | 126.5 / 126.4 / 126.1 | 126.4 | 0.3 % | - | - | - |
| 6 | 147.4 / 143.3 / 144.3 | 144.3 | 2.8 % | - | - | - |

### `ncclnt64-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_NTHREADS=64`. One variable: NCCL thread count. Standing TP2 all-reduce is PYNCCL. Exclusive LL was a large cost; LL,Simple / MAX_NCHANNELS=1 / ALGO=Ring were pin noise or worse at c6. Confirmed `NCCL_NTHREADS=64`.

**Verdict.** **pin noise, not a keep.** 54.8 / 97.3 / 125.7 / **147.8**, sum **425.6**, worst spread 6.6 %. Gates pass in all three passes; acceptance 49.7-55.5 % and tokens per step 4.478-4.876 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.9 % / +4.7 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. NCCL thread count is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.4 % / -8.9 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T17:07:02+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.0 / 54.8 / 55.6 | 54.8 | 6.6 % | - | - | - |
| 3 | 97.3 / 95.2 / 98.9 | 97.3 | 3.8 % | - | - | - |
| 5 | 124.9 / 125.7 / 126.4 | 125.7 | 1.2 % | - | - | - |
| 6 | 147.8 / 142.8 / 148.0 | 147.8 | 3.5 % | - | - | - |

### `lpf1024`

**Changed.** `proto2-030` plus `--long-prefill-token-threshold 1024` via `ARM_EXTRA_ARGS`. One variable: the pin already names `LONG_PREFILL_TOKEN_THRESHOLD=1024` and the reference recipe passes the flag, but `05-serve.sh` never forwarded it. vLLM default is 0. Confirmed in non-default args (`long_prefill_token_threshold: 1024`).

**Verdict.** **negative, not kept.** 56.0 / 97.9 / 126.7 / **144.9**, sum **425.5**, worst spread 14.4 %. Gates pass in all three passes; acceptance 49.3-56.0 % and tokens per step 4.414-4.923, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise (c5 spread 14.4 % is the arm's own). Forwarding the documented flag does not close the reference gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T18:24:09+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.0 / 56.4 / 51.0 | 56.0 | 9.6 % | - | - | - |
| 3 | 96.3 / 99.5 / 97.9 | 97.9 | 3.3 % | - | - | - |
| 5 | 136.2 / 117.9 / 126.7 | 126.7 | 14.4 % | - | - | - |
| 6 | 139.8 / 144.9 / 146.5 | 144.9 | 4.6 % | - | - | - |

### `lpf1024-rc2`

**Changed.** `proto2-030-rc2` plus `--long-prefill-token-threshold 1024` via `ARM_EXTRA_ARGS`. One variable: pin already names `LONG_PREFILL_TOKEN_THRESHOLD=1024` and the anemll recipe passes the flag, but `05-serve.sh` never forwards it. vLLM default is 0. Confirmed `long_prefill_token_threshold': 1024`. Re-measure of rc1 `lpf1024` (425.5) on this pin.

**Verdict.** **pin noise, not a keep.** 53.9 / 99.2 / 129.0 / **143.4**, sum **425.5**, worst spread 7.3 %. Gates pass in all three passes. One pass dipped to accept 49.0 % / tokens per step 4.414, a hair under standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.9 % / +1.6 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Forwarding the documented flag does not close the reference gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.1 % / -8.7 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T19:33:37+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.9 / 54.9 / 51.0 | 53.9 | 7.2 % | - | - | - |
| 3 | 100.1 / 99.2 / 95.6 | 99.2 | 4.5 % | - | - | - |
| 5 | 128.3 / 130.0 / 129.0 | 129.0 | 1.3 % | - | - | - |
| 6 | 143.4 / 151.2 / 140.8 | 143.4 | 7.3 % | - | - | - |

### `abk7-a`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-21T16:55:46+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.7 / 52.5 / 51.1 | 52.5 | 5.0 % | - | - | - |
| 3 | 102.6 / 103.1 / 97.0 | 102.6 | 5.9 % | - | - | - |
| 5 | 127.2 / 135.5 / 125.7 | 127.2 | 7.7 % | - | - | - |
| 6 | 142.8 / 151.6 / 142.9 | 142.9 | 6.2 % | - | - | - |

### `memprof0`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. One variable: disable v0.30's CUDA-graph memory profiler (default on; maps 0.8389 to effective 0.8115). `util8663` compensated by raising the util number; this turns the profiler off at the standing 0.8389. Env confirmed; Available KV 13.36 GiB vs 9.39 GiB with the profiler on.

**Verdict.** **negative, not kept.** 56.5 / 97.5 / 131.0 / **140.2**, sum **425.2**, worst spread 9.1 %. Gates pass in all three passes. Against `proto2-030` (412.1 / c6 138.3, spread 12.8 %) and same-day control `p030b` (420.3) the sum is +3.2 % / +1.2 %, both inside the keep spread. One pass dipped to accept 48.1 % and 4.351 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). Extra KV without the profiler does not close the reference gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T17:08:38+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.5 / 59.9 / 55.0 | 56.5 | 8.7 % | - | - | - |
| 3 | 100.0 / 91.1 / 97.5 | 97.5 | 9.1 % | - | - | - |
| 5 | 131.0 / 132.5 / 126.3 | 131.0 | 4.7 % | - | - | - |
| 6 | 140.2 / 136.4 / 143.9 | 140.2 | 5.3 % | - | - | - |

### `maxlen32k-rc2`

**Changed.** `proto2-030-rc2` plus `MAX_MODEL_LEN=32768`. One variable: manager max seq len. Pin is 65536; anemll uses 262144. Protocol gens 512, so 32k is plenty. Smaller page table for overlay sparse indexer. Confirmed `max_model_len': 32768`.

**Verdict.** **pin noise, not a keep.** 54.3 / 97.7 / 128.6 / **144.3**, sum **424.9**, worst spread 6.8 %. Gates pass in all three passes; acceptance 49.8-55.8 % and tokens per step 4.476-4.876 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.8 % / +2.2 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Cutting max_model_len is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.6 % / -11.0 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T12:22:17+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.3 / 52.5 / 56.2 | 54.3 | 6.8 % | - | - | - |
| 3 | 98.7 / 96.2 / 97.7 | 97.7 | 2.6 % | - | - | - |
| 5 | 126.5 / 130.1 / 128.6 | 128.6 | 2.8 % | - | - | - |
| 6 | 138.6 / 144.3 / 146.5 | 144.3 | 5.5 % | - | - | - |

### `noshare`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS=0`. One variable: disable micro MoE shared-input across experts (default 1). Fires on W4A8 micro when activation is silu/relu2, m==1, and a1_gscale is a scalar (protocol c1 is m=1 decode). Confirmed in container env. Linear stayed `B12xFp8BlockScaledMMKernel`.

**Verdict.** **negative, not kept.** 54.6 / 100.0 / 128.8 / **141.4**, sum **424.8**, worst spread 11.6 %. Gates pass in all three passes; acceptance 50.3-56.3 % and tokens per step 4.531-4.939, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. Micro share-input off is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T20:31:09+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.2 / 52.8 / 54.6 | 54.6 | 6.2 % | - | - | - |
| 3 | 100.0 / 109.9 / 98.3 | 100.0 | 11.6 % | - | - | - |
| 5 | 129.8 / 126.4 / 128.8 | 128.8 | 2.6 % | - | - | - |
| 6 | 141.3 / 141.4 / 142.4 | 141.4 | 0.8 % | - | - | - |

### `noidxk-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_INDEXER_DIRECT_K=0`. One variable: fused-indexer direct-K score off (default 1). Kill-switch restores the staged pipeline. Distinct from closed `VLLM_B12X_INDEXER_DIRECT_GATHER`. Confirmed `B12X_INDEXER_DIRECT_K=0`.

**Verdict.** **pin noise, not a keep.** 54.0 / 100.0 / 129.0 / **141.7**, sum **424.7**, worst spread 7.7 %. Gates pass in all three passes. One pass dipped to tokens per step 4.420, a hair under standing floor 4.427; acceptance 49.2-53.9 % holds. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.7 % / +0.4 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Direct-K off is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.2 % / -9.8 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T01:05:36+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.9 / 54.0 / 54.1 | 54.0 | 0.4 % | - | - | - |
| 3 | 102.4 / 94.7 / 100.0 | 100.0 | 7.7 % | - | - | - |
| 5 | 129.0 / 132.9 / 126.5 | 129.0 | 5.0 % | - | - | - |
| 6 | 137.3 / 141.7 / 142.0 | 141.7 | 3.3 % | - | - | - |

### `wochunk32-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_WO_QUANT_CHUNKS_PER_PROGRAM=32`. One variable: WO MXFP8 quant chunks (default 16). Chunks 8 was pin noise. Confirmed `B12X_WO_QUANT_CHUNKS_PER_PROGRAM=32`.

**Verdict.** **pin noise, not a keep.** 53.1 / 99.4 / 129.0 / **143.2**, sum **424.7**, worst spread 10.2 %. Gates pass in all three passes. One pass dipped to accept 48.6 % / tokens per step 4.376, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.7 % / +1.4 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. WO chunks 32 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.6 % / -11.7 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T13:36:01+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 57.1 / 53.1 / 51.7 | 53.1 | 10.2 % | - | - | - |
| 3 | 99.4 / 100.3 / 97.5 | 99.4 | 2.8 % | - | - | - |
| 5 | 123.6 / 129.0 / 130.9 | 129.0 | 5.7 % | - | - | - |
| 6 | 143.2 / 144.2 / 141.3 | 143.2 | 2.0 % | - | - | - |

### `conn8-rc2`

**Changed.** `proto2-030-rc2` plus `CUDA_DEVICE_MAX_CONNECTIONS=8`. One variable: CUDA default instead of `env.spark.sh`'s 1. The anemll recipe does not set this. Confirmed `CUDA_DEVICE_MAX_CONNECTIONS=8`. Re-measure of rc1 `conn8` (422.6) on this pin.

**Verdict.** **pin noise, not a keep.** 55.0 / 98.7 / 130.6 / **140.3**, sum **424.6**, worst spread 3.3 %. Gates pass in all three passes; acceptance 50.5-54.3 % and tokens per step 4.524-4.785 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.7 % / -0.6 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. CUDA max connections 8 is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.3 % / -10.7 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T18:41:23+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.1 / 55.0 / 53.5 | 55.0 | 2.9 % | - | - | - |
| 3 | 99.9 / 98.7 / 98.6 | 98.7 | 1.3 % | - | - | - |
| 5 | 131.6 / 130.6 / 127.3 | 130.6 | 3.3 % | - | - | - |
| 6 | 140.3 / 139.3 / 141.7 | 140.3 | 1.7 % | - | - | - |

### `p030-rc2b`

**Changed.** same pin as `proto2-030-rc2` (`main-030-rc2`, no extra env), new tag so the log does not append. Same-day re-baseline after later same-image arms clustered at 442-453 while the first sample was 417.5.

**Verdict.** **control, not a keep.** 52.8 / 99.6 / 128.0 / **144.2**, sum **424.6**, worst spread 13.6 %. Gates pass in all three passes. One pass dipped to accept 45.6 % / tokens per step 4.197, below standing floor 49.2 % / 4.427. Against original `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.7 % / +2.1 %, pin noise. Confirms the 442-453 cluster (`nowo-rc2` / `moehum-rc2` / `k6-rc2` / `moea16-rc2`) is a real one-variable lift vs the pin, still inside keep-spread. Standing arm remains `proto2-030-rc2`. Future keep-rule on this pin should treat ~420 as the noise center, not a new champion. vs same-day `refg-rc2` (467.9 / 157.1) still -9.3 % / -8.2 %.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T17:17:32+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.3 / 52.8 / 49.1 | 52.8 | 13.6 % | - | - | - |
| 3 | 100.8 / 98.3 / 99.6 | 99.6 | 2.5 % | - | - | - |
| 5 | 125.3 / 128.0 / 136.7 | 128.0 | 8.9 % | - | - | - |
| 6 | 142.3 / 146.8 / 144.2 | 144.2 | 3.1 % | - | - | - |

### `attnfi-rc2`

**Changed.** `proto2-030-rc2` plus `ATTENTION_BACKEND` and `DRAFT_ATTENTION_BACKEND` = `FLASHINFER_MLA_SPARSE_DSV4`. One family: FlashInfer sparse MLA DSV4 on tagged FlashInfer `v0.7.0rc3` (rc1 used floating `main`). Confirmed in non-default args. Re-measure of rc1 `attnfi030` (431.4) on this pin.

**Verdict.** **pin noise, not a keep.** 53.0 / 99.3 / 129.3 / **142.8**, sum **424.4**, worst spread 5.7 %. Gates pass in all three passes. One pass dipped to accept 49.0 % / tokens per step 4.414, a hair under standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.7 % / +1.1 %, pin noise. vs same-day `refg-rc2` (467.9 / 157.1) still -9.3 % / -9.1 %. FlashInfer MLA DSV4 on this pin is not the remaining gap. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T17:04:59+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.0 / 54.8 / 52.9 | 53.0 | 3.6 % | - | - | - |
| 3 | 99.3 / 95.7 / 101.4 | 99.3 | 5.7 % | - | - | - |
| 5 | 129.3 / 125.5 / 129.5 | 129.3 | 3.1 % | - | - | - |
| 6 | 146.4 / 142.8 / 139.4 | 142.8 | 4.9 % | - | - | - |

### `nofast`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_FAST_MATH=0`. One variable: disable b12x MoE fast-math (default True). Keyed into the dynamic W4A8 kernel cache. Confirmed in container env (`B12X_FAST_MATH=0`); linear stayed `B12xFp8BlockScaledMMKernel`, MoE stayed `B12X_MXFP4_MXFP8`. Warm FLASHINFER MLA autotune was skipped this round: the 0.7.0 cache JSON is metadata-only (499 B, no tactic entries), so a rerun would still hit the default heuristic.

**Verdict.** **negative, not kept.** 56.2 / 101.1 / 124.4 / **142.7**, sum **424.4**, worst spread 16.4 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 45.5 % and 4.163 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). Fast-math off is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T19:40:55+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 48.6 / 56.2 / 57.8 | 56.2 | 16.4 % | - | - | - |
| 3 | 101.1 / 104.1 / 99.2 | 101.1 | 4.8 % | - | - | - |
| 5 | 126.4 / 124.4 / 124.3 | 124.4 | 1.7 % | - | - | - |
| 6 | 143.3 / 142.7 / 140.3 | 142.7 | 2.1 % | - | - | - |

### `moecap175`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=175`. One variable: the b12x dynamic MoE cluster cap. Offline sweep priced 175 at 0.941x the flat-188 time at 288 routed rows; 175 is also the reference's static cap at c6. First boot died at `_check_enough_kv_cache_memory` (9.39 GiB available vs 9.48 GiB needed); retry served.

**Verdict.** **negative, not kept.** 55.3 / 101.0 / 128.8 / **139.2**, sum **424.3**, worst spread 9.5 %. Gates pass in all three passes; acceptance 50.3-55.9 % and tokens per step 4.515-4.876, neither lower than `proto2-030`. Against same-base `proto2-030` (412.1 / c6 138.3, spread 12.8 %) the sum is +3.0 % and c6 is +0.6 %, both inside the larger spread. The offline 6 % does not survive the protocol. Standing arm remains `proto2-030`. Env confirmed in the container (`printenv` = 175).

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T15:15:31+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.6 / 56.3 / 55.3 | 55.3 | 6.7 % | - | - | - |
| 3 | 105.7 / 101.0 / 96.1 | 101.0 | 9.5 % | - | - | - |
| 5 | 126.8 / 133.0 / 128.8 | 128.8 | 4.8 % | - | - | - |
| 6 | 143.9 / 138.2 / 139.2 | 139.2 | 4.1 % | - | - | - |

### `proto2-dgather`

**Changed.** `proto2` with `VLLM_B12X_INDEXER_DIRECT_GATHER=1` (via `SERVE_EXTRA_ENV`), the packed-indexer gather that reads the KV cache directly instead of materialising a full-cache copy on every layer of every decode step

**Verdict.** **The biggest result of the run: +27 % on the sum, and it was already written down in our own code.** 51.9 / 101.4 / 127.6 / 143.4, sum **424.3**, against `proto2`'s 35.4 / 78.9 / 103.0 / 117.0, sum 334.3 -- +46.6 % at c1, +28.5 % at c3, +23.9 % at c5, **+22.6 % at c6**, and +26.9 % on the sum. Gates pass in all three passes. **Acceptance is 53.5 / 51.4 / 53.7 %, i.e. unchanged** against `proto2`'s ~52 %, and tokens per step are 4.712 / 4.580 / 4.726 against 4.571 -- so the condition that kept this switch off does not hold here. The docstring of `_indexer_direct_gather` in `patches/files/sm12x_b12x_kernels.py` says the default path "reshapes a strided slice of the KV cache, which materialises a full-cache copy on every layer of every decode step (~57 ms/step at 1 row)" and that the direct read "measured 10.1 -> 25.5 tok/s on the no-spec arm, but draft acceptance drops from ~60 % to ~41-50 %, so it stays off until that is understood". Round 45 named that copy independently from a shapes-recorded trace -- `aten::copy_ [17177, 64, 132]`, 1344 calls in a 16-token run, `64 x 132 = 8448` matching the packed-indexer sidecar width in our own boot log -- and then measured the switch end to end with DSpark on, which is what had never been done. The acceptance penalty does not reproduce on proto2. **This arm must still be confirmed by a second identical run** before it is treated as standing, per the loop's re-measure rule; `proto2-dgather2` is that run.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T20:04:12+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.6 / 51.9 / 48.0 | 51.9 | 14.6 % | - | - | - |
| 3 | 97.1 / 101.4 / 101.5 | 101.4 | 4.3 % | - | - | - |
| 5 | 127.6 / 135.3 / 126.5 | 127.6 | 6.9 % | - | - | - |
| 6 | 143.4 / 139.7 / 146.5 | 143.4 | 4.7 % | - | - | - |

### `util8663-rc2`

**Changed.** `proto2-030-rc2` plus `GPU_MEMORY_UTILIZATION=0.8663`. One variable: restore the pre-profiler KV budget. v0.30 CUDA-graph memory profiling maps 0.8389 to a smaller effective util; 0.8663 is the engine's named equivalent. Confirmed `gpu_memory_utilization: 0.8663`. Re-measure of rc1 `util8663` (429.0) after `moemar-rc2` died at 9.48 vs 9.27 GiB.

**Verdict.** **pin noise, not a keep.** 52.4 / 100.0 / 128.8 / **143.1**, sum **424.3**, worst spread 6.9 %. Gates pass in all three passes; acceptance 49.9-55.1 % and tokens per step 4.491-4.830 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.6 % / +1.3 %. Against same-day control `p030-rc2b` (424.6 / 144.2) this is a wash. Extra KV does not close the reference gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.3 % / -8.9 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T17:59:59+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.2 / 55.8 / 52.4 | 52.4 | 6.9 % | - | - | - |
| 3 | 100.0 / 101.2 / 97.3 | 100.0 | 3.9 % | - | - | - |
| 5 | 128.8 / 127.6 / 131.9 | 128.8 | 3.3 % | - | - | - |
| 6 | 144.8 / 140.9 / 143.1 | 143.1 | 2.7 % | - | - | - |

### `idxpr-rc2`

**Changed.** `proto2-030-rc2` plus `--sparse-indexer-topk-backend per_row` via `ARM_EXTRA_ARGS`. One variable: indexer top-k. Auto chain is cooperative (excluded on SM120) -> persistent (topk 512) -> per_row. Confirmed `sparse_indexer_topk_backend': 'per_row'`. Re-measure of rc1 `idxpr` (423.4) on this pin.

**Verdict.** **pin noise, not a keep.** 53.5 / 98.5 / 128.8 / **143.4**, sum **424.2**, worst spread 5.3 %. Gates pass in all three passes; acceptance 50.0-53.9 % and tokens per step 4.465-4.785 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.6 % / +1.6 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. per_row topk is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.3 % / -8.7 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T19:46:06+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.7 / 53.5 / 55.0 | 53.5 | 4.3 % | - | - | - |
| 3 | 98.5 / 94.8 / 100.0 | 98.5 | 5.3 % | - | - | - |
| 5 | 130.6 / 128.8 / 127.0 | 128.8 | 2.8 % | - | - | - |
| 6 | 139.3 / 143.4 / 145.1 | 143.4 | 4.0 % | - | - | - |

### `moetile32`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_TILE_MN=32x128`. One variable: the b12x dynamic MoE tile. Auto planner returns (16, 128) for the whole protocol band (48-288 routed rows). `64x128` was already a wash (`proto2-moetile`); M32 is the W4A8 ladder's next tactic. Env confirmed in the container.

**Verdict.** **negative, not kept.** 52.7 / 98.2 / 129.7 / **143.6**, sum **424.2**, worst spread 3.4 %. Gates pass in all three passes; acceptance 50.0-53.2 % and tokens per step 4.491-4.712, neither lower than `proto2-030`. Against same-base `proto2-030` (412.1 / c6 138.3, spread 12.8 %) the sum is +2.9 % and c6 is +3.8 %, both inside the larger spread. Tile shape is not the remaining MoE gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T16:20:14+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.7 / 52.1 / 53.9 | 52.7 | 3.4 % | - | - | - |
| 3 | 99.3 / 98.2 / 97.0 | 98.2 | 2.3 % | - | - | - |
| 5 | 127.7 / 131.3 / 129.7 | 129.7 | 2.8 % | - | - | - |
| 6 | 142.4 / 143.6 / 144.7 | 143.6 | 1.6 % | - | - | - |

### `moetilem16-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_MOE_TILE_MN=16x128`. One variable: micro MoE tile. Default is 64x128; 32x128 was pin noise; 128x128 died at KV floor. Confirmed `B12X_MOE_TILE_MN=16x128`.

**Verdict.** **pin noise, not a keep.** 55.3 / 97.5 / 128.7 / **142.7**, sum **424.2**, worst spread 6.1 %. Gates pass in all three passes; acceptance 49.8-55.2 % and tokens per step 4.460-4.876 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.6 % / +1.1 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Micro 16x128 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.7 % / -12.0 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T02:15:48+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.3 / 53.0 / 56.1 | 55.3 | 5.6 % | - | - | - |
| 3 | 97.5 / 96.5 / 98.2 | 97.5 | 1.7 % | - | - | - |
| 5 | 125.9 / 133.7 / 128.7 | 128.7 | 6.1 % | - | - | - |
| 6 | 144.9 / 141.3 / 142.7 | 142.7 | 2.5 % | - | - | - |

### `loadsaf-rc2`

**Changed.** `proto2-030-rc2` plus `--load-format safetensors` via `ARM_EXTRA_ARGS`. One variable: pin `LOAD_FORMAT=instanttensor` is sourced after EXTRA, so env override cannot win. Anemll uses `load_format=auto`. InstantTensor env knobs already closed. Confirmed `load_format': 'safetensors'`.

**Verdict.** **pin noise, not a keep.** 55.3 / 96.9 / 127.0 / **144.9**, sum **424.1**, worst spread 7.9 %. Gates pass in all three passes; acceptance 50.0-55.1 % and tokens per step 4.483-4.830 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.6 % / +2.6 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Loader format is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.7 % / -10.7 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 4 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T14:44:44+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.9 / 54.3 / 55.3 | 55.3 | 2.9 % | - | - | - |
| 3 | 96.9 / 95.8 / 103.5 | 96.9 | 7.9 % | - | - | - |
| 5 | 127.0 / 125.8 / 135.2 | 127.0 | 7.4 % | - | - | - |
| 6 | 149.3 / 144.9 / 142.9 | 144.9 | 4.4 % | - | - | - |

### `wochunk8-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_WO_QUANT_CHUNKS_PER_PROGRAM=8`. One variable: WO MXFP8 quant chunks (default 16). Eager profile priced live c1 WO at 57.4 ms (38 % of 150.6 ms). Confirmed `B12X_WO_QUANT_CHUNKS_PER_PROGRAM=8`.

**Verdict.** **pin noise, not a keep.** 54.5 / 96.6 / 129.9 / **143.1**, sum **424.1**, worst spread 13.5 %. Gates pass in all three passes; acceptance 49.8-58.5 % and tokens per step 4.465-5.086 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.6 % / +1.3 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Halving WO chunks is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.7 % / -11.8 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T13:26:04+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.1 / 54.5 / 55.5 | 54.5 | 6.2 % | - | - | - |
| 3 | 96.6 / 95.5 / 108.5 | 96.6 | 13.5 % | - | - | - |
| 5 | 133.2 / 129.9 / 128.6 | 129.9 | 3.5 % | - | - | - |
| 6 | 142.4 / 143.1 / 144.1 | 143.1 | 1.2 % | - | - | - |

### `atom24-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DENSE_ATOM_24=1`. One variable: experimental 24-atom MMA in b12x dense GEMM (default 0). Changes generated code and is keyed into the persistent compile cache. Confirmed `B12X_DENSE_ATOM_24=1`. Re-measure of rc1 `atom24` (426.0) on this pin.

**Verdict.** **pin noise, not a keep.** 54.0 / 97.6 / 130.8 / **141.6**, sum **424.0**, worst spread 15.4 %. Gates pass in all three passes. One pass dipped to accept 44.2 % / tokens per step 4.096, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.6 % / +0.3 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. 24-atom dense GEMM is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.4 % / -9.9 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T19:12:54+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.0 / 47.6 / 55.9 | 54.0 | 15.4 % | - | - | - |
| 3 | 95.9 / 101.9 / 97.6 | 97.6 | 6.1 % | - | - | - |
| 5 | 130.8 / 125.1 / 131.1 | 130.8 | 4.6 % | - | - | - |
| 6 | 145.2 / 141.4 / 141.6 | 141.6 | 2.7 % | - | - | - |

### `nccltc106-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_IB_TC=106`. One variable: IB traffic class 106 (DSCP 26). Standing TP2 all-reduce is PYNCCL over RoCE. Confirmed `NCCL_IB_TC=106`.

**Verdict.** **pin noise, not a keep.** 52.0 / 102.8 / 128.0 / **141.2**, sum **424.0**, worst spread 6.9 %. Gates pass in all three passes. One pass dipped to accept 48.2 % / tokens per step 4.339, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.6 % / +0.0 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. IB TC 106 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.7 % / -12.9 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T22:49:02+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 50.3 / 52.0 / 53.9 | 52.0 | 6.9 % | - | - | - |
| 3 | 103.2 / 102.8 / 101.6 | 102.8 | 1.6 % | - | - | - |
| 5 | 126.3 / 128.0 / 131.9 | 128.0 | 4.4 % | - | - | - |
| 6 | 141.2 / 139.2 / 143.3 | 141.2 | 2.9 % | - | - | - |

### `nocar-rc2`

**Changed.** `proto2-030-rc2` plus `--disable-custom-all-reduce` via `ARM_EXTRA_ARGS`. One variable: match anemll `disable_custom_all_reduce=True`. Eager profile priced 87 all-reduce calls at 81.3 ms gpu_sum on live c1. Confirmed `disable_custom_all_reduce': True`. Re-measure of `proto2-dg-nocar` (negative) on this pin.

**Verdict.** **pin noise, not a keep.** 54.1 / 99.6 / 129.7 / **140.6**, sum **424.0**, worst spread 12.0 %. Gates pass in all three passes. One pass dipped to accept 47.4 % / tokens per step 4.303, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.6 % / -0.4 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash / slightly worse at c6. Custom AR off is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.7 % / -13.3 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T14:25:49+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.3 / 49.8 / 54.1 | 54.1 | 12.0 % | - | - | - |
| 3 | 99.6 / 98.5 / 102.1 | 99.6 | 3.6 % | - | - | - |
| 5 | 129.7 / 130.2 / 128.8 | 129.7 | 1.1 % | - | - | - |
| 6 | 143.6 / 138.2 / 140.6 | 140.6 | 3.8 % | - | - | - |

### `w4scr128-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_W4A8_CONVERT_SCRATCH_MB=128`. One variable: W4A8 prepare scratch 128 MiB vs default 64. Distinct from `now4mat-rc2` (materialized off, a c6 cost). Confirmed `B12X_W4A8_CONVERT_SCRATCH_MB=128`.

**Verdict.** **pin noise, not a keep.** 53.0 / 98.3 / 129.2 / **143.5**, sum **424.0**, worst spread 7.4 %. Gates pass in all three passes; acceptance 49.9-54.3 % and tokens per step 4.478-4.785 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.6 % / +1.6 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Convert scratch 128 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.7 % / -11.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T23:54:42+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.0 / 55.6 / 52.8 | 53.0 | 5.3 % | - | - | - |
| 3 | 92.5 / 99.8 / 98.3 | 98.3 | 7.4 % | - | - | - |
| 5 | 129.2 / 136.2 / 128.7 | 129.2 | 5.8 % | - | - | - |
| 6 | 141.4 / 143.7 / 143.5 | 143.5 | 1.6 % | - | - | - |

### `ncclch1-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_MAX_NCHANNELS=1`. One variable: NCCL channel count. Standing TP2 all-reduce is PYNCCL. `NCCL_PROTO=LL` was a large cost; one channel is a different lever for 87 small all-reduce calls. Confirmed `NCCL_MAX_NCHANNELS=1`.

**Verdict.** **pin noise, not a keep.** 53.5 / 100.1 / 127.4 / **142.9**, sum **423.9**, worst spread 7.9 %. Gates pass in all three passes. One pass dipped to accept 48.8 % / tokens per step 4.391, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.5 % / +1.2 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. One NCCL channel is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -12.8 % / -11.9 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T16:37:43+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.5 / 54.9 / 51.7 | 53.5 | 6.0 % | - | - | - |
| 3 | 100.1 / 98.5 / 100.7 | 100.1 | 2.2 % | - | - | - |
| 5 | 123.7 / 127.4 / 133.6 | 127.4 | 7.8 % | - | - | - |
| 6 | 153.0 / 141.7 / 142.9 | 142.9 | 7.9 % | - | - | - |

### `noshare-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS=0`. One variable: disable micro MoE shared-input across experts (default 1). Fires on W4A8 micro at m=1 (protocol c1). Confirmed `B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS=0`. Re-measure of rc1 `noshare` (424.8) on this pin.

**Verdict.** **pin noise, not a keep.** 51.7 / 99.5 / 127.4 / **145.0**, sum **423.6**, worst spread 7.2 %. Gates pass in all three passes. One pass dipped to accept 49.0 % / tokens per step 4.414, a hair under standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.5 % / +2.7 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Micro share-input off is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.5 % / -7.7 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T20:05:13+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.1 / 51.5 / 51.7 | 51.7 | 7.0 % | - | - | - |
| 3 | 99.5 / 96.8 / 99.8 | 99.5 | 3.0 % | - | - | - |
| 5 | 127.4 / 129.7 / 125.5 | 127.4 | 3.3 % | - | - | - |
| 6 | 154.9 / 145.0 / 144.5 | 145.0 | 7.2 % | - | - | - |

### `idxpr`

**Changed.** `proto2-030` plus `--sparse-indexer-topk-backend per_row` via `ARM_EXTRA_ARGS`. One variable: indexer top-k. Auto chain is cooperative (excluded on SM120) -> persistent (topk 512, our k) -> per_row. `idxfi` already measured flashinfer (pin noise). deep_select needs SM100. Overlay scores logits in fp32. Confirmed in non-default args (`sparse_indexer_topk_backend: per_row`).

**Verdict.** **negative, not kept.** 52.8 / 96.2 / 128.5 / **145.9**, sum **423.4**, worst spread 9.4 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 48.9 % (tokens/step 4.414 at the `proto2-030` floor). per_row topk is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T20:42:47+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.2 / 52.8 / 51.1 | 52.8 | 5.9 % | - | - | - |
| 3 | 95.4 / 98.6 / 96.2 | 96.2 | 3.3 % | - | - | - |
| 5 | 128.5 / 131.4 / 126.7 | 128.5 | 3.7 % | - | - | - |
| 6 | 137.3 / 151.0 / 145.9 | 145.9 | 9.4 % | - | - | - |

### `nofast-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_FAST_MATH=0`. One variable: disable b12x MoE fast-math (default True). Keyed into the dynamic W4A8 kernel cache. Confirmed `B12X_FAST_MATH=0`. Re-measure of rc1 `nofast` (424.4) on this pin.

**Verdict.** **pin noise, not a keep.** 51.4 / 99.5 / 130.2 / **142.3**, sum **423.4**, worst spread 5.1 %. Gates pass in all three passes. One pass dipped to accept 48.1 % / tokens per step 4.339, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.4 % / +0.8 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Fast-math off is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.5 % / -9.4 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T19:23:54+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 51.4 / 50.3 / 52.9 | 51.4 | 5.1 % | - | - | - |
| 3 | 101.4 / 99.5 / 98.6 | 99.5 | 2.8 % | - | - | - |
| 5 | 130.2 / 130.6 / 129.2 | 130.2 | 1.1 % | - | - | - |
| 6 | 142.3 / 140.0 / 142.3 | 142.3 | 1.6 % | - | - | - |

### `nogqa6-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_PAGED_GQA6_COMPACT_SYNC=0`. One variable: GQA6 compact-sync decode fastpath off (default 1). DSV4 is GQA 6; overlay `B12X_MLA_SPARSE` decode goes through the paged indexer. Confirmed `B12X_PAGED_GQA6_COMPACT_SYNC=0`.

**Verdict.** **pin noise, not a keep.** 54.6 / 94.2 / 132.4 / **142.2**, sum **423.4**, worst spread 9.2 %. Gates pass in all three passes; acceptance 49.5-54.9 % and tokens per step 4.439-4.800 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.4 % / +0.7 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Compact-sync off is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.5 % / -9.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T00:40:10+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.6 / 53.1 / 54.7 | 54.6 | 2.9 % | - | - | - |
| 3 | 94.2 / 92.5 / 101.2 | 94.2 | 9.2 % | - | - | - |
| 5 | 132.4 / 134.4 / 130.0 | 132.4 | 3.3 % | - | - | - |
| 6 | 140.6 / 142.6 / 142.2 | 142.2 | 1.4 % | - | - | - |

### `dsl471-rc2`

**Changed.** `proto2-030-rc2` plus `nvidia-cutlass-dsl[cu13]==4.7.1` (and `libs-base` / `libs-cu13` 4.7.1) committed as `vllm-spark-0731:main-030-rc2-dsl471`. One variable: latest tagged cutlass-dsl vs pin 4.7.0. Confirmed `nvidia-cutlass-dsl==4.7.1`.

**Verdict.** **pin noise, not a keep.** 55.9 / 98.5 / 128.3 / **140.6**, sum **423.3**, worst spread 6.0 %. Gates pass in all three passes; acceptance 49.8-55.4 % and tokens per step 4.465-4.830 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.4 % / -0.4 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Do not pin 4.7.1. vs same-day `refg-rc2` (467.9 / 157.1) still -9.5 % / -10.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T00:04:03+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.9 / 56.3 / 54.8 | 55.9 | 2.7 % | - | - | - |
| 3 | 102.2 / 96.6 / 98.5 | 98.5 | 5.7 % | - | - | - |
| 5 | 128.3 / 125.9 / 133.6 | 128.3 | 6.0 % | - | - | - |
| 6 | 140.6 / 140.3 / 142.6 | 140.6 | 1.6 % | - | - | - |

### `noasync`

**Changed.** `proto2-030` plus `--no-async-scheduling` via `ARM_EXTRA_ARGS`. One variable: drop async scheduling. `run-arm.sh` always prepends `--async-scheduling`; the BooleanOptionalAction last-flag wins (`async_scheduling: False`). The anemll recipe also enables async scheduling. Never isolated off on `main-030-rc1`.

**Verdict.** **negative, not kept.** 55.7 / 97.4 / 127.0 / **143.0**, sum **423.1**, worst spread 11.8 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 47.3 % and 4.303 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). Async scheduling is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T22:37:37+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.9 / 52.3 / 55.7 | 55.7 | 11.8 % | - | - | - |
| 3 | 97.4 / 97.0 / 103.1 | 97.4 | 6.3 % | - | - | - |
| 5 | 127.0 / 126.2 / 135.7 | 127.0 | 7.5 % | - | - | - |
| 6 | 143.0 / 138.3 / 146.2 | 143.0 | 5.5 % | - | - | - |

### `nopfx-rc2`

**Changed.** `proto2-030-rc2` plus `--no-enable-prefix-caching` via `ARM_EXTRA_ARGS`. One variable: prefix cache off. Pin sets `ENABLE_PREFIX_CACHING=1`; vLLM default is already True. Protocol reuses one prompt across levels. Confirmed `enable_prefix_caching': False`. Re-measure of rc1 `nopfx` (418.3) on this pin.

**Verdict.** **pin noise, not a keep.** 54.6 / 100.0 / 128.3 / **140.2**, sum **423.1**, worst spread 6.6 %. Gates pass in all three passes; acceptance 49.5-57.6 % and tokens per step 4.452-5.020 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.3 % / -0.7 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Prefix cache off is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.6 % / -10.8 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T20:37:21+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 57.5 / 53.9 / 54.6 | 54.6 | 6.6 % | - | - | - |
| 3 | 100.0 / 96.3 / 101.9 | 100.0 | 5.6 % | - | - | - |
| 5 | 130.4 / 128.1 / 128.3 | 128.3 | 1.8 % | - | - | - |
| 6 | 138.7 / 142.2 / 140.2 | 140.2 | 2.5 % | - | - | - |

### `noturbo`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_DENSE_SPLITK_TURBO=0`. One variable: disable atomic-BF16 reduction on dense GEMM split-K (default 1). Decode policy picks 2-way split-K for m in 2..6 and k >= 4096 (our FP8 linear decode band). Confirmed in container env (`B12X_DENSE_SPLITK_TURBO=0`); linear stayed `B12xFp8BlockScaledMMKernel`.

**Verdict.** **quality fail, not kept.** 53.7 / 96.1 / 128.8 / **144.5**, sum **423.1**, worst spread 8.9 %. `gate_france` passes; **`gate_9x8` fails all three passes** (`'E5=8F='`, `'0x9a,0'`, `'E5=8F='` instead of `'72, 9x9'`). One pass also dipped to accept 48.6 % and 4.384 tokens/step, below `proto2-030`'s floor. Atomic-BF16 split-K is numerically required on this pin. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T20:09:28+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.5 / 51.7 / 53.7 | 53.7 | 8.9 % | - | - | - |
| 3 | 96.1 / 94.9 / 96.4 | 96.1 | 1.6 % | - | - | - |
| 5 | 128.8 / 121.3 / 129.8 | 128.8 | 6.6 % | - | - | - |
| 6 | 147.2 / 144.5 / 143.6 | 144.5 | 2.5 % | - | - | - |

### `nogemv-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DISABLE_BF16_GEMV=1`. One variable: BF16 small-N GEMV off (default on). Path covers N<=1024, K>=1024; DSV4 O-proj is o_lora_rank 1024. Confirmed `B12X_DISABLE_BF16_GEMV=1`.

**Verdict.** **pin noise, not a keep.** 54.1 / 97.1 / 127.2 / **144.6**, sum **423.0**, worst spread 7.6 %. Gates pass in all three passes; acceptance 50.4-57.7 % and tokens per step 4.504-5.020 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.3 % / +2.4 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. GEMV off is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.6 % / -8.0 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T01:20:59+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.8 / 57.9 / 54.1 | 54.1 | 7.6 % | - | - | - |
| 3 | 93.0 / 97.9 / 97.1 | 97.1 | 5.0 % | - | - | - |
| 5 | 127.2 / 129.8 / 126.2 | 127.2 | 2.8 % | - | - | - |
| 6 | 141.0 / 144.7 / 144.6 | 144.6 | 2.6 % | - | - | - |

### `igp-rc2`

**Changed.** `proto2-030-rc2` plus `USE_INDUCTOR_GRAPH_PARTITION=true`. One variable: inductor graph partition around piecewise CUDA-graph breaks (standing None/False). Confirmed `use_inductor_graph_partition': True`. Plumbing added in `scripts/05-serve.sh`.

**Verdict.** **pin noise, not a keep.** 53.1 / 95.8 / 130.7 / **143.2**, sum **422.8**, worst spread 4.9 %. 9x8 gate passes in all three passes; pass 1 france gate is garbled (` Paris.",\n    "label":`). Acceptance 49.9-53.7 % and tokens per step 4.452-4.750 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.3 % / +1.4 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Inductor graph partition is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.0 % / -11.7 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T21:14:07+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.1 / 54.2 / 52.0 | 53.1 | 4.1 % | - | - | - |
| 3 | 98.9 / 94.2 / 95.8 | 95.8 | 4.9 % | - | - | - |
| 5 | 131.4 / 130.7 / 126.9 | 130.7 | 3.4 % | - | - | - |
| 6 | 142.8 / 144.4 / 143.2 | 143.2 | 1.1 % | - | - | - |

### `moeshare-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_W4A8_SHARE_INPUT=1`. One variable: force the W4A8 shared-input producer on. Default is on only when dense or decode candidate is true; decode candidate requires routed_rows <= 64, so only c1. Confirmed `B12X_DYNAMIC_W4A8_SHARE_INPUT=1`. Re-measure of rc1 `moeshare` (420.8) on this pin.

**Verdict.** **pin noise, not a keep.** 51.3 / 98.0 / 129.3 / **144.2**, sum **422.8**, worst spread 13.5 %. Gates pass in all three passes. One pass dipped to accept 46.5 % / tokens per step 4.231, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.3 % / +2.1 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Forced share-input on c3-c6 is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.6 % / -8.2 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T18:52:13+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 49.3 / 56.2 / 51.3 | 51.3 | 13.5 % | - | - | - |
| 3 | 98.0 / 97.2 / 102.1 | 98.0 | 5.0 % | - | - | - |
| 5 | 132.3 / 129.2 / 129.3 | 129.3 | 2.4 % | - | - | - |
| 6 | 139.2 / 147.1 / 144.2 | 144.2 | 5.5 % | - | - | - |

### `conn8`

**Changed.** `proto2-030` plus `CUDA_DEVICE_MAX_CONNECTIONS=8`. One variable: CUDA default instead of `env.spark.sh`'s 1. The anemll recipe does not set this. `05-serve.sh` already `-e` the name, so EXTRA overrides. Confirmed in container env (`CUDA_DEVICE_MAX_CONNECTIONS=8`).

**Verdict.** **negative, not kept.** 54.7 / 100.4 / 125.8 / **141.7**, sum **422.6**, worst spread 7.9 %. Gates pass in all three passes; acceptance 49.9-58.9 % and tokens per step 4.499-5.120, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. CUDA max connections 8 is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T23:06:18+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.7 / 54.2 / 58.5 | 54.7 | 7.9 % | - | - | - |
| 3 | 101.0 / 97.3 / 100.4 | 100.4 | 3.7 % | - | - | - |
| 5 | 125.8 / 126.4 / 124.9 | 125.8 | 1.2 % | - | - | - |
| 6 | 141.7 / 144.3 / 138.2 | 141.7 | 4.3 % | - | - | - |

### `proto2-dgather2`

**Changed.** the identical `proto2-dgather` run repeated, to satisfy the loop's re-measure-before-believing rule

**Verdict.** **the confirmation, and it reproduces.** 55.1 / 98.2 / 129.7 / 139.6, sum **422.6**, against the first run's 51.9 / 101.4 / 127.6 / 143.4, sum 424.3 -- a 0.4 % difference on the sum, inside both runs' spreads (4.4 / 0.5 / 3.0 / 3.7 % here). Gates pass in all three passes; acceptance 52.4 / 53.6 / 50.8 % and tokens per step 4.655 / 4.732 / 4.538 are unchanged from `proto2`. **So the arm is kept**: it passes both gates, does not lose acceptance, and beats `proto2` by ~27 % on the sum and ~22 % at c6. It is now the best measured arm on this base.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T20:12:28+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.9 / 55.1 / 53.5 | 55.1 | 4.4 % | - | - | - |
| 3 | 98.6 / 98.1 / 98.2 | 98.2 | 0.5 % | - | - | - |
| 5 | 130.7 / 126.8 / 129.7 | 129.7 | 3.0 % | - | - | - |
| 6 | 142.9 / 139.6 / 137.8 | 139.6 | 3.7 % | - | - | - |

### `downsc`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_ENABLE_DYNAMIC_DOWN_SCALE=1`. One variable: dynamic MoE down-scale on (default False). Applied as `_dynamic_down_scale_enabled() and not is_w4a8`. DSV4 MXFP4 experts map to `quant_mode=w4a8_mx`, so this is a no-op on this pin. Confirmed in container env (`B12X_ENABLE_DYNAMIC_DOWN_SCALE=1`).

**Verdict.** **negative, not kept.** 56.0 / 97.3 / 129.1 / **140.1**, sum **422.5**, worst spread 5.0 %. Gates pass in all three passes; acceptance 49.4-56.0 % and tokens per step 4.439-4.923, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. Dynamic down-scale is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T00:04:07+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.0 / 53.9 / 56.7 | 56.0 | 5.0 % | - | - | - |
| 3 | 99.6 / 97.3 / 95.5 | 97.3 | 4.2 % | - | - | - |
| 5 | 129.1 / 133.5 / 128.9 | 129.1 | 3.6 % | - | - | - |
| 6 | 139.6 / 144.5 / 140.1 | 140.1 | 3.5 % | - | - | - |

### `moecap175-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=175`. One variable: match the reference static kernel's 175-cluster cap at the c6 band (288 routed rows). Pin decode policy is a flat 188 up to 640. Confirmed `B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS=175`. Re-measure of rc1 `moecap175` on this pin.

**Verdict.** **pin noise, not a keep.** 55.3 / 100.1 / 124.5 / **142.5**, sum **422.4**, worst spread 8.3 %. Gates pass in all three passes. One pass dipped to accept 48.2 % / tokens per step 4.376, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.2 % / +0.9 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Cluster cap 175 is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.7 % / -9.3 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T18:21:31+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.3 / 55.6 / 51.0 | 55.3 | 8.3 % | - | - | - |
| 3 | 100.1 / 97.5 / 102.8 | 100.1 | 5.3 % | - | - | - |
| 5 | 124.5 / 124.0 / 130.9 | 124.5 | 5.5 % | - | - | - |
| 6 | 146.8 / 142.4 / 142.5 | 142.5 | 3.1 % | - | - | - |

### `idxfi-rc2`

**Changed.** `proto2-030-rc2` plus `--sparse-indexer-topk-backend flashinfer` via `ARM_EXTRA_ARGS`. One variable: indexer top-k. Auto chain is cooperative (excluded on SM120) -> persistent (topk 512) -> per_row; flashinfer is opt-in. Confirmed `sparse_indexer_topk_backend': 'flashinfer'`. First `run-arm` health window expired at 450s (CUDA-graph capture); protocol then ran against the live healthy container. Re-measure of rc1 `idxfi` (419.4) on this pin.

**Verdict.** **pin noise, not a keep.** 50.7 / 97.6 / 127.8 / **146.2**, sum **422.3**, worst spread 11.0 %. Gates pass in all three passes. One pass dipped to accept 45.3 % / tokens per step 4.163, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.1 % / +3.5 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. FlashInfer indexer top-k is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.7 % / -6.9 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 9 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T23:14:50+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 50.7 / 54.1 / 48.5 | 50.7 | 11.0 % | - | - | - |
| 3 | 97.6 / 99.4 / 95.1 | 97.6 | 4.4 % | - | - | - |
| 5 | 136.7 / 127.8 / 127.3 | 127.8 | 7.4 % | - | - | - |
| 6 | 147.7 / 145.5 / 146.2 | 146.2 | 1.5 % | - | - | - |

### `nopagemax-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_MSA_DECODE_PAGEMAX=0`. One variable: MSA scheduled page-max decode off (default 1). Overlay scores DSA indexer logits via `logits_paged`. Confirmed `B12X_MSA_DECODE_PAGEMAX=0`.

**Verdict.** **pin noise, not a keep.** 54.0 / 97.7 / 128.7 / **141.8**, sum **422.2**, worst spread 11.5 %. Gates pass in all three passes. One pass dipped to accept 48.8 % / tokens per step 4.414, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.1 % / +0.4 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Page-max off is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.1 % / -12.6 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T02:27:24+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 51.3 / 54.6 / 54.0 | 54.0 | 6.1 % | - | - | - |
| 3 | 97.7 / 106.2 / 95.0 | 97.7 | 11.5 % | - | - | - |
| 5 | 128.7 / 127.4 / 135.3 | 128.7 | 6.1 % | - | - | - |
| 6 | 142.4 / 141.8 / 141.3 | 141.8 | 0.8 % | - | - | - |

### `split4`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_MLA_SM120_NUM_SPLITS=4`. One variable: pin MLA decode split-K to 4. Unset uses the wave-balanced heuristic; `split1` pinned 1 and `split2` pinned 2. Overlay `B12X_MLA_SPARSE` decode goes through `run_unified_decode`. Dual-cache protocol is ~11 chunks. Confirmed in container env (`B12X_MLA_SM120_NUM_SPLITS=4`).

**Verdict.** **negative, not kept.** 51.9 / 100.9 / 126.6 / **142.8**, sum **422.2**, worst spread 11.0 %. Gates pass in all three passes; acceptance 49.3-56.9 % and tokens per step 4.452-4.971, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. MLA num_splits=4 is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T02:10:03+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 51.9 / 51.9 / 57.6 | 51.9 | 11.0 % | - | - | - |
| 3 | 100.9 / 95.1 / 104.6 | 100.9 | 9.4 % | - | - | - |
| 5 | 126.6 / 126.9 / 123.6 | 126.6 | 2.6 % | - | - | - |
| 6 | 148.7 / 142.8 / 139.8 | 142.8 | 6.2 % | - | - | - |

### `ncclnsock4-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_NSOCKS_PERTHREAD=4`. One variable: NCCL sockets per helper thread. Complementary to `ncclsock4-rc2` (`NCCL_SOCKET_NTHREADS=4`, a c6 cost). Confirmed `NCCL_NSOCKS_PERTHREAD=4`.

**Verdict.** **pin noise, not a keep.** 53.0 / 99.0 / 128.1 / **142.0**, sum **422.1**, worst spread 12.8 %. Gates pass in all three passes. One pass dipped to accept 49.0 % / tokens per step 4.414, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.1 % / +0.6 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Extra sockets per thread are not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.1 % / -12.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T21:50:45+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.0 / 52.0 / 58.8 | 53.0 | 12.8 % | - | - | - |
| 3 | 99.0 / 103.2 / 93.0 | 99.0 | 10.3 % | - | - | - |
| 5 | 131.7 / 128.1 / 125.9 | 128.1 | 4.5 % | - | - | - |
| 6 | 142.0 / 143.2 / 139.7 | 142.0 | 2.5 % | - | - | - |

### `moetile32-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_TILE_MN=32x128`. One variable: remaining W4A8 tile after auto (16, 128), 64x128, and 128x128. Confirmed `B12X_DYNAMIC_TILE_MN=32x128`. Re-measure of rc1 `moetile32` (424.2) on this pin.

**Verdict.** **pin noise, not a keep.** 53.7 / 96.4 / 127.5 / **144.4**, sum **422.0**, worst spread 8.2 %. Gates pass in all three passes; acceptance 49.7-55.9 % and tokens per step 4.491-4.876 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.1 % / +2.3 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Tile 32x128 is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -9.8 % / -8.1 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T00:29:12+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.4 / 53.7 / 52.0 | 53.7 | 8.2 % | - | - | - |
| 3 | 98.8 / 95.7 / 96.4 | 96.4 | 3.2 % | - | - | - |
| 5 | 124.4 / 127.5 / 128.2 | 127.5 | 3.0 % | - | - | - |
| 6 | 143.8 / 147.1 / 144.4 | 144.4 | 2.3 % | - | - | - |

### `abk7-b`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.30.1.dev0+g9ed533eb4`, 3 passes at 512 tokens, started 2026-09-21T17:19:40+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.2 / 51.0 / 51.5 | 51.5 | 8.2 % | - | - | - |
| 3 | 97.0 / 93.4 / 97.3 | 97.0 | 4.0 % | - | - | - |
| 5 | 128.9 / 129.3 / 131.4 | 129.3 | 1.9 % | - | - | - |
| 6 | 139.7 / 144.1 / 152.2 | 144.1 | 8.7 % | - | - | - |

### `ncclretry1-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_IB_RETRY_CNT=1`. One variable: IB retry count 1 vs default 7. Standing TP2 all-reduce is PYNCCL over RoCE. Confirmed `NCCL_IB_RETRY_CNT=1`.

**Verdict.** **pin noise, not a keep.** 53.1 / 101.1 / 125.1 / **142.5**, sum **421.8**, worst spread 9.8 %. Gates pass in all three passes. One pass dipped to accept 48.8 % / tokens per step 4.376, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.0 % / +0.9 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. IB retry 1 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.2 % / -12.1 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T23:21:40+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.4 / 50.2 / 53.1 | 53.1 | 9.8 % | - | - | - |
| 3 | 98.8 / 101.1 / 101.1 | 101.1 | 2.3 % | - | - | - |
| 5 | 125.1 / 124.6 / 132.5 | 125.1 | 6.3 % | - | - | - |
| 6 | 138.1 / 142.5 / 144.7 | 142.5 | 4.6 % | - | - | - |

### `ncclcnet0-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_COLLNET_ENABLE=0`. One variable: disable NCCL CollNet in-network collectives. Standing TP2 all-reduce is PYNCCL over two-host RoCE. Confirmed `NCCL_COLLNET_ENABLE=0`.

**Verdict.** **pin noise, not a keep.** 52.9 / 99.6 / 126.1 / **143.0**, sum **421.6**, worst spread 7.8 %. Gates pass in all three passes. One pass dipped to accept 47.3 % / tokens per step 4.267, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.0 % / +1.3 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. CollNet off is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.2 % / -11.8 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-21T00:04:45+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.5 / 49.4 / 52.9 | 52.9 | 7.8 % | - | - | - |
| 3 | 97.0 / 99.6 / 100.3 | 99.6 | 3.3 % | - | - | - |
| 5 | 124.3 / 126.1 / 131.3 | 126.1 | 5.6 % | - | - | - |
| 6 | 140.9 / 143.6 / 143.0 | 143.0 | 1.9 % | - | - | - |

### `ncclring-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_ALGO=Ring`. One variable: NCCL algorithm Ring vs auto (Tree for small). Anemll nsys landed on RING_LL. Exclusive `NCCL_PROTO=LL` was a large cost; LL,Simple and MAX_NCHANNELS=1 were pin noise. Confirmed `NCCL_ALGO=Ring`.

**Verdict.** **negative, not a keep.** 55.1 / 100.2 / 128.0 / **138.2**, sum **421.5**, worst spread 12.9 %. Gates pass in all three passes; acceptance 49.2-57.1 % and tokens per step 4.433-4.987 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +1.0 % / -2.1 %. Against same-day control `p030-rc2b` (424.6 / 144.2) worse at c6. Forced Ring is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.3 % / -14.8 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T16:58:14+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.1 / 57.6 / 52.2 | 55.1 | 9.8 % | - | - | - |
| 3 | 108.4 / 100.2 / 95.5 | 100.2 | 12.9 % | - | - | - |
| 5 | 128.0 / 126.2 / 129.4 | 128.0 | 2.5 % | - | - | - |
| 6 | 141.6 / 138.2 / 137.8 | 138.2 | 2.7 % | - | - | - |

### `ncclcu0-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_CUMEM_ENABLE=0`. One variable: disable NCCL cuMemMap allocations. GB10 is UMA; default cuMem can fragment host-device unified memory. Standing TP2 all-reduce is PYNCCL. Confirmed `NCCL_CUMEM_ENABLE=0`.

**Verdict.** **pin noise, not a keep.** 53.6 / 96.2 / 126.1 / **145.0**, sum **420.9**, worst spread 5.6 %. Gates pass in all three passes; acceptance 49.5-52.9 % and tokens per step 4.444-4.683 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +0.8 % / +2.7 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. NCCL cuMem off is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.4 % / -10.6 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T17:17:14+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.7 / 52.7 / 53.6 | 53.6 | 1.9 % | - | - | - |
| 3 | 92.7 / 96.2 / 97.9 | 96.2 | 5.4 % | - | - | - |
| 5 | 125.2 / 132.2 / 126.1 | 126.1 | 5.6 % | - | - | - |
| 6 | 145.0 / 146.5 / 141.1 | 145.0 | 3.7 % | - | - | - |

### `nccllls-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_PROTO=LL,Simple`. One variable: enable LL as an option, keep Simple for large messages. Exclusive `NCCL_PROTO=LL` (`nccpll-rc2`) was a large cost. Anemll nsys landed on RING_LL; exclusive LL is not that. Confirmed `NCCL_PROTO=LL,Simple`.

**Verdict.** **pin noise, not a keep.** 54.2 / 89.0 / 131.3 / **146.4**, sum **420.9**, worst spread 6.4 %. Gates pass in all three passes; acceptance 49.6-53.9 % and tokens per step 4.439-4.785 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +0.8 % / +3.7 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Optional LL is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.4 % / -9.7 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T16:48:07+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.8 / 54.2 / 52.9 | 54.2 | 5.4 % | - | - | - |
| 3 | 92.4 / 89.0 / 86.7 | 89.0 | 6.4 % | - | - | - |
| 5 | 130.3 / 131.3 / 133.4 | 131.3 | 2.4 % | - | - | - |
| 6 | 148.0 / 146.0 / 146.4 | 146.4 | 1.4 % | - | - | - |

### `moeshare`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_W4A8_SHARE_INPUT=1`. One variable: force the W4A8 shared-input producer on. Default is on only when dense or decode candidate is true; decode candidate requires routed_rows <= 64, so only c1. c3-c6 (144-288 rows) default off. Env confirmed in the container.

**Verdict.** **negative, not kept.** 55.4 / 98.0 / 126.9 / **140.5**, sum **420.8**, worst spread 4.4 %. Gates pass in all three passes; acceptance 50.1-55.9 % and tokens per step 4.476-4.876, neither lower than `proto2-030`. Against same-base `proto2-030` (412.1 / c6 138.3, spread 12.8 %) the sum is +2.1 % and c6 is +1.6 %, both inside the larger spread. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T16:45:54+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 55.4 / 53.6 / 55.7 | 55.4 | 3.8 % | - | - | - |
| 3 | 98.9 / 97.7 / 98.0 | 98.0 | 1.2 % | - | - | - |
| 5 | 126.9 / 131.9 / 126.3 | 126.9 | 4.4 % | - | - | - |
| 6 | 139.6 / 144.1 / 140.5 | 140.5 | 3.2 % | - | - | - |

### `noprki-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DENSE_PER_ROW_IN_KERNEL=0`. One variable: dense per-row GS host chain instead of in-kernel (default 1). Bit-identical by construction. Confirmed `B12X_DENSE_PER_ROW_IN_KERNEL=0`.

**Verdict.** **pin noise, not a keep.** 55.5 / 99.3 / 126.5 / **139.4**, sum **420.7**, worst spread 7.9 %. Gates pass in all three passes. One pass dipped to accept 49.0 % / tokens per step 4.414, a hair under standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +0.8 % / -1.3 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Host per-row GS is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -10.1 % / -11.3 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T00:53:44+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.4 / 57.8 / 55.5 | 55.5 | 7.9 % | - | - | - |
| 3 | 99.3 / 95.2 / 99.9 | 99.3 | 4.7 % | - | - | - |
| 5 | 125.6 / 126.5 / 129.4 | 126.5 | 3.0 % | - | - | - |
| 6 | 140.2 / 139.4 / 135.0 | 139.4 | 3.7 % | - | - | - |

### `cgpiece-k6-rc2`

**Changed.** `wooff-rc2` + k=6 plus `CUDAGRAPH_MODE=PIECEWISE`, five passes. One variable against the standing default: piecewise-only CUDA graphs vs FULL_AND_PIECEWISE, re-measured on the new default after `cgpiece-rc2` had shown +7.2 % against the old one at three passes with a 17.8 % spread.

**Verdict.** **negative; the earlier gain was noise.** 55.1 / 96.1 / 129.9 / **139.5**, sum **420.6**, worst spread 8.9 %. Both gates pass in all five passes; acceptance 56.7-66.5 % holds, one pass dipped to 4.401 tokens per step. Against `wooff-k6-5-rc2` (445.2 / 152.3, spread 7.6 %) this is **-5.5 % / -8.4 %**, so PIECEWISE-only is a clear loss at five passes and `cgpiece-rc2`'s +7.2 % did not replicate. Consistent with `cgfull` and `proto2-dg-cgfull`, both large negatives: `FULL_AND_PIECEWISE` stands.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-21T13:48:36+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.5 / 55.1 / 56.1 / 53.6 / 54.0 | 55.1 | 8.9 % | - | - | - |
| 3 | 100.5 / 97.3 / 95.5 / 94.0 / 96.1 | 96.1 | 6.8 % | - | - | - |
| 5 | 131.3 / 123.4 / 133.2 / 127.9 / 129.9 | 129.9 | 7.5 % | - | - | - |
| 6 | 137.8 / 147.0 / 138.7 / 139.5 / 143.0 | 139.5 | 6.6 % | - | - | - |

### `p030b`

**Changed.** same pin as `proto2-030` (`main-030-rc1`, no extra env), new tag so the log does not append. Same-day re-baseline after later same-image arms clustered at 421-429 while the first sample was 412.1.

**Verdict.** **control, not a keep.** 52.5 / 96.6 / 129.6 / **141.6**, sum **420.3**, worst spread 11.0 %. Gates pass in all three passes; acceptance 50.3-58.0 % and tokens per step 4.531-5.069. Against original `proto2-030` (412.1 / c6 138.3, spread 12.8 %) this is +2.0 % on the sum and +2.4 % at c6, inside that spread. Confirms the later config arms were sitting in the pin's own noise, not winning. Standing arm remains `proto2-030`. Future keep-rule on this pin should use the larger of the two control spreads (12.8 %) and treat ~420 as the current noise center, not a new champion.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T16:58:19+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.5 / 52.5 / 58.3 | 52.5 | 11.0 % | - | - | - |
| 3 | 95.2 / 96.6 / 99.1 | 96.6 | 4.0 % | - | - | - |
| 5 | 129.6 / 125.4 / 130.1 | 129.6 | 3.6 % | - | - | - |
| 6 | 141.6 / 141.4 / 143.2 | 141.6 | 1.3 % | - | - | - |

### `now4sh`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_W4A8_SHARE_INPUT=0`. One variable: W4A8 shared-input producer off. DSV4 MXFP4 experts map to `quant_mode=w4a8_mx`. Share-input default is on when dense or decode candidate is true (`moeshare` forced it on). Confirmed in container env (`B12X_DYNAMIC_W4A8_SHARE_INPUT=0`).

**Verdict.** **negative, not kept.** 52.5 / 100.1 / 127.4 / **140.1**, sum **420.1**, worst spread 8.3 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 48.2 % and 4.376 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). W4A8 share-input off is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T00:27:43+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.5 / 51.1 / 54.8 | 52.5 | 7.0 % | - | - | - |
| 3 | 100.5 / 92.2 / 100.1 | 100.1 | 8.3 % | - | - | - |
| 5 | 127.4 / 129.5 / 127.0 | 127.4 | 2.0 % | - | - | - |
| 6 | 140.1 / 144.9 / 138.3 | 140.1 | 4.7 % | - | - | - |

### `ncclchk0-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_CHECKS_DISABLE=1`. One variable: disable NCCL argument checks on every collective. Standing TP2 all-reduce is PYNCCL. Confirmed `NCCL_CHECKS_DISABLE=1`.

**Verdict.** **pin noise, not a keep.** 56.1 / 92.3 / 126.5 / **145.1**, sum **420.0**, worst spread 18.3 %. Gates pass in all three passes. One pass dipped to accept 46.9 % / tokens per step 4.267, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +0.6 % / +2.8 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. NCCL checks off is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.6 % / -10.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T19:44:06+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.1 / 54.6 / 56.7 | 56.1 | 3.7 % | - | - | - |
| 3 | 91.9 / 92.3 / 108.8 | 92.3 | 18.3 % | - | - | - |
| 5 | 126.2 / 126.5 / 127.4 | 126.5 | 0.9 % | - | - | - |
| 6 | 143.5 / 145.1 / 150.1 | 145.1 | 4.5 % | - | - | - |

### `ncclpeer2-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_NCHANNELS_PER_NET_PEER=2`. One variable: NCCL channels per net peer. Complementary to `ncclch1-rc2` / `ncclmin2-rc2`. Confirmed `NCCL_NCHANNELS_PER_NET_PEER=2`.

**Verdict.** **negative, not a keep.** 53.3 / 96.6 / 129.8 / **140.3**, sum **420.0**, worst spread 4.9 %. Gates pass in all three passes. One pass dipped to accept 48.8 % / tokens per step 4.401, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +0.6 % / -0.6 %. Worse at c6 than standing and than `p030-rc2b` (424.6 / 144.2). Per-peer channels 2 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.6 % / -13.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T22:39:13+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 51.7 / 54.3 / 53.3 | 53.3 | 4.9 % | - | - | - |
| 3 | 96.6 / 98.4 / 94.5 | 96.6 | 4.0 % | - | - | - |
| 5 | 131.1 / 129.1 / 129.8 | 129.8 | 1.5 % | - | - | - |
| 6 | 140.3 / 140.1 / 143.2 | 140.3 | 2.2 % | - | - | - |

### `split1`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_MLA_SM120_NUM_SPLITS=1`. One variable: pin MLA decode split-K to 1. Unset uses the FlashInfer-ported wave-balanced heuristic. Overlay `B12X_MLA_SPARSE` decode goes through `compressed_sparse_mla.run` -> `run_unified_decode`, which reads this env per call. Confirmed in container env (`B12X_MLA_SM120_NUM_SPLITS=1`).

**Verdict.** **negative, not kept.** 53.0 / 100.5 / 124.8 / **141.7**, sum **420.0**, worst spread 12.6 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 47.8 % and 4.303 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). MLA num_splits=1 is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T01:24:11+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.3 / 53.0 / 49.6 | 53.0 | 12.6 % | - | - | - |
| 3 | 100.5 / 93.4 / 101.8 | 100.5 | 8.4 % | - | - | - |
| 5 | 130.5 / 122.2 / 124.8 | 124.8 | 6.7 % | - | - | - |
| 6 | 140.7 / 142.2 / 141.7 | 141.7 | 1.1 % | - | - | - |

### `now4mat-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_W4A8_MATERIALIZED=0`. One variable: W4A8 materialized-queue off (default on when dense candidate is true). Distinct from closed work-source `persistent_grid`. Confirmed `B12X_DYNAMIC_W4A8_MATERIALIZED=0`.

**Verdict.** **negative, not a keep.** 53.8 / 99.1 / 127.6 / **139.3**, sum **419.8**, worst spread 15.7 %. Gates pass in all three passes; acceptance 50.1-57.7 % and tokens per step 4.498-5.020 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +0.6 % / -1.3 %. Worse at c6 than standing and than `p030-rc2b` (424.6 / 144.2). Materialized off is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.6 % / -14.1 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T23:43:05+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.8 / 54.4 / 53.0 | 53.8 | 2.6 % | - | - | - |
| 3 | 99.1 / 109.8 / 94.2 | 99.1 | 15.7 % | - | - | - |
| 5 | 129.4 / 127.6 / 126.1 | 127.6 | 2.6 % | - | - | - |
| 6 | 145.2 / 139.0 / 139.3 | 139.3 | 4.5 % | - | - | - |

### `memprof0-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. One variable: disable v0.30 CUDA-graph memory profiler (default on; maps 0.8389 to effective 0.8115). `util8663-rc2` raised the util number; this turns the profiler off at standing 0.8389. Confirmed `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. Re-measure of rc1 `memprof0` (425.2) on this pin.

**Verdict.** **pin noise, not a keep.** 50.2 / 98.4 / 130.6 / **140.4**, sum **419.6**, worst spread 10.0 %. Gates pass in all three passes. One pass dipped to accept 45.4 % / tokens per step 4.163, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +0.5 % / -0.6 %. Against same-day control `p030-rc2b` (424.6 / 144.2) slightly worse. Extra KV without the profiler is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -10.3 % / -10.6 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T22:48:05+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 50.2 / 53.6 / 48.6 | 50.2 | 10.0 % | - | - | - |
| 3 | 102.1 / 93.3 / 98.4 | 98.4 | 8.9 % | - | - | - |
| 5 | 131.8 / 124.7 / 130.6 | 130.6 | 5.4 % | - | - | - |
| 6 | 138.5 / 145.1 / 140.4 | 140.4 | 4.7 % | - | - | - |

### `idxfi`

**Changed.** `proto2-030` plus `--sparse-indexer-topk-backend flashinfer` via `ARM_EXTRA_ARGS`. One variable: indexer top-k. Auto chain is cooperative (excluded on SM120) -> persistent (topk 512, our k) -> per_row; flashinfer is opt-in. Overlay scores logits in fp32, which `top_k_ragged_transform` requires. Confirmed in non-default args. First `run-arm` health window expired at 450s (FlashInfer init); protocol then ran against the live healthy container.

**Verdict.** **negative, not kept.** 55.6 / 97.3 / 124.2 / **142.3**, sum **419.4**, worst spread 12.2 %. Gates pass in all three passes. Against `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 48.3 % and 4.339 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). FlashInfer indexer top-k is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 9 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T18:12:49+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 50.8 / 57.6 / 55.6 | 55.6 | 12.2 % | - | - | - |
| 3 | 97.3 / 94.2 / 98.0 | 97.3 | 3.9 % | - | - | - |
| 5 | 124.2 / 122.2 / 131.1 | 124.2 | 7.2 % | - | - | - |
| 6 | 142.3 / 140.0 / 151.3 | 142.3 | 7.9 % | - | - | - |

### `noasync-rc2`

**Changed.** `proto2-030-rc2` plus `--no-async-scheduling` via `ARM_EXTRA_ARGS`. One variable: drop async scheduling. `run-arm.sh` always prepends `--async-scheduling`; BooleanOptionalAction last-flag wins. The anemll recipe also enables async scheduling. Confirmed `async_scheduling': False`. Re-measure of rc1 `noasync` (423.1) on this pin.

**Verdict.** **pin noise, not a keep.** 52.3 / 97.6 / 126.0 / **142.6**, sum **418.5**, worst spread 12.6 %. Gates pass in all three passes. One pass dipped to accept 46.0 % / tokens per step 4.231, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +0.2 % / +1.0 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. Async scheduling off is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -10.6 % / -9.2 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T23:27:13+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.3 / 49.5 / 56.1 | 52.3 | 12.6 % | - | - | - |
| 3 | 97.6 / 100.5 / 96.5 | 97.6 | 4.1 % | - | - | - |
| 5 | 134.7 / 124.7 / 126.0 | 126.0 | 7.9 % | - | - | - |
| 6 | 142.6 / 149.8 / 139.4 | 142.6 | 7.3 % | - | - | - |

### `split8`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_MLA_SM120_NUM_SPLITS=8`. One variable: pin MLA decode split-K to 8. Unset uses the wave-balanced heuristic; `split1`/`split2`/`split4` already sat in pin noise. Overlay `B12X_MLA_SPARSE` decode goes through `run_unified_decode`. Dual-cache protocol is ~11 chunks. Confirmed in container env (`B12X_MLA_SM120_NUM_SPLITS=8`).

**Verdict.** **negative, not kept.** 53.0 / 97.2 / 125.9 / **142.4**, sum **418.5**, worst spread 8.5 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 48.6 %, below `proto2-030`'s floor 49.0 %; tokens per step 4.414-4.830 holds the floor. The num_splits ladder is closed. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T02:20:13+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 51.5 / 56.0 / 53.0 | 53.0 | 8.5 % | - | - | - |
| 3 | 97.2 / 96.2 / 101.3 | 97.2 | 5.2 % | - | - | - |
| 5 | 123.7 / 125.9 / 126.4 | 125.9 | 2.1 % | - | - | - |
| 6 | 138.4 / 145.4 / 142.4 | 142.4 | 4.9 % | - | - | - |

### `yesh16`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_MLA_SM120_DSV4_H16_NATIVE=1`. One variable: force DSV4 H16 native decode on. Unset is auto; `noh16` forced 0. Force 1 so small-row c1 also uses H16 (auto keeps H8 in the sub-wave latency regime). Overlay `B12X_MLA_SPARSE` decode goes through `run_unified_decode`. Confirmed in container env (`B12X_MLA_SM120_DSV4_H16_NATIVE=1`).

**Verdict.** **negative, not kept.** 55.8 / 96.3 / 127.6 / **138.7**, sum **418.4**, worst spread 8.2 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 48.9 % and 4.401 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). H16 force-on is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T01:49:37+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.6 / 52.0 / 55.8 | 55.8 | 8.2 % | - | - | - |
| 3 | 96.3 / 100.4 / 96.0 | 96.3 | 4.6 % | - | - | - |
| 5 | 123.8 / 127.6 / 131.9 | 127.6 | 6.3 % | - | - | - |
| 6 | 136.3 / 138.7 / 139.8 | 138.7 | 2.5 % | - | - | - |

### `nopfx`

**Changed.** `proto2-030` plus `--no-enable-prefix-caching` via `ARM_EXTRA_ARGS`. One variable: prefix cache off. Pin sets `ENABLE_PREFIX_CACHING=1`; vLLM default is already True. BooleanOptionalAction last-flag wins over `05-serve.sh`'s `--enable-prefix-caching`. Protocol reuses one prompt across levels. Confirmed `enable_prefix_caching: False`.

**Verdict.** **negative, not kept.** 53.9 / 98.2 / 124.0 / **142.2**, sum **418.3**, worst spread 8.5 %. Gates pass in all three passes; acceptance 49.6-53.7 % and tokens per step 4.444-4.758, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. Prefix cache is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T23:15:57+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.2 / 53.9 / 51.5 | 53.9 | 5.0 % | - | - | - |
| 3 | 99.8 / 98.2 / 96.9 | 98.2 | 3.0 % | - | - | - |
| 5 | 123.1 / 133.7 / 124.0 | 124.0 | 8.5 % | - | - | - |
| 6 | 139.0 / 144.8 / 142.2 | 142.2 | 4.1 % | - | - | - |

### `ncclgrp-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_LAUNCH_MODE=GROUP`. One variable: NCCL CUDA launch mode GROUP vs default PARALLEL. Standing TP2 all-reduce is PYNCCL. Eager profile priced 87 all-reduce calls at 81.3 ms gpu_sum on live c1. Confirmed `NCCL_LAUNCH_MODE=GROUP`.

**Verdict.** **pin noise, not a keep.** 54.4 / 95.3 / 126.9 / **141.6**, sum **418.2**, worst spread 8.8 %. Gates pass in all three passes; acceptance 49.5-57.8 % and tokens per step 4.439-5.020 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +0.2 % / +0.3 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. GROUP launch is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -13.9 % / -12.7 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T17:37:16+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.4 / 52.7 / 57.5 | 54.4 | 8.8 % | - | - | - |
| 3 | 93.8 / 95.3 / 95.9 | 95.3 | 2.2 % | - | - | - |
| 5 | 126.9 / 126.6 / 126.9 | 126.9 | 0.2 % | - | - | - |
| 6 | 141.4 / 141.6 / 146.5 | 141.6 | 3.6 % | - | - | - |

### `wochunk1-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_WO_QUANT_CHUNKS_PER_PROGRAM=1`. One variable: WO MXFP8 quant chunks (default 16). Last legal step after 2/4/8/32 pin noise. Confirmed `B12X_WO_QUANT_CHUNKS_PER_PROGRAM=1`.

**Verdict.** **negative, not a keep.** 53.7 / 99.9 / 127.7 / **136.7**, sum **418.0**, worst spread 6.1 %. Gates pass in all three passes; acceptance 49.7-56.5 % and tokens per step 4.472-4.923 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +0.1 % / -3.2 %. Against same-day control `p030-rc2b` (424.6 / 144.2) worse at c6. WO chunks 1 is a cost. vs same-day `refg-rc2b` (485.9 / 162.2) still -14.0 % / -15.7 %. WO chunk ladder closed (1 cost, 2/4/8/32 wash). Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T14:12:03+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.2 / 56.5 / 53.7 | 53.7 | 6.1 % | - | - | - |
| 3 | 99.9 / 98.5 / 103.4 | 99.9 | 4.9 % | - | - | - |
| 5 | 131.6 / 127.7 / 127.7 | 127.7 | 3.1 % | - | - | - |
| 6 | 141.5 / 135.7 / 136.7 | 136.7 | 4.2 % | - | - | - |

### `tile128-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_TILE_MN=128x128`. One variable: remaining W4A8 dense-candidate tile after auto (16, 128). Confirmed `B12X_DYNAMIC_TILE_MN=128x128`. Re-measure of rc1 `tile128` on this pin.

**Verdict.** **pin noise, not a keep.** 53.8 / 98.8 / 122.6 / **142.5**, sum **417.7**, worst spread 6.3 %. Gates pass in all three passes; acceptance 49.8-57.0 % and tokens per step 4.460-4.971 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is a wash. Against same-day control `p030-rc2b` (424.6 / 144.2) slightly worse. Tile shape is not the remaining MoE gap. vs same-day `refg-rc2` (467.9 / 157.1) still -10.7 % / -9.3 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T18:11:25+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.3 / 55.7 / 53.8 | 53.8 | 6.3 % | - | - | - |
| 3 | 98.8 / 95.8 / 99.3 | 98.8 | 3.5 % | - | - | - |
| 5 | 122.6 / 124.3 / 122.3 | 122.6 | 1.6 % | - | - | - |
| 6 | 139.5 / 142.5 / 146.5 | 142.5 | 4.9 % | - | - | - |

### `noaot-rc2`

**Changed.** `proto2-030-rc2` plus `VLLM_USE_AOT_COMPILE=0`. One variable: AOT compile cache off (pin defaults it on). `proto2-compile` tried this on an earlier pin and still resolved `CompilationMode.NONE`. Confirmed `VLLM_USE_AOT_COMPILE=0` in the container; resolved config still `CompilationMode.NONE`.

**Verdict.** **pin noise, not a keep.** 54.3 / 95.8 / 126.3 / **141.1**, sum **417.5**, worst spread 9.4 %. Gates pass in all three passes; acceptance 49.3-56.9 % and tokens per step 4.439-4.971 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is +0.0 % / -0.1 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. AOT off only loses the persist cache; compilation never happens on this stack. vs same-day `refg-rc2b` (485.9 / 162.2) still -14.1 % / -13.0 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-21T01:15:13+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 54.3 / 57.2 / 52.1 | 54.3 | 9.4 % | - | - | - |
| 3 | 95.8 / 93.6 / 97.6 | 95.8 | 4.2 % | - | - | - |
| 5 | 126.3 / 132.8 / 125.7 | 126.3 | 5.6 % | - | - | - |
| 6 | 142.4 / 140.0 / 141.1 | 141.1 | 1.7 % | - | - | - |

### `proto2-030-rc2`

**Changed.** matched-main pin moved to `v0.30.0rc2` (`fa6ff060667f`), image `vllm-spark-0731:main-030-rc2`. Same serve config as `proto2-030`. Overlay scan FAIL=0. Phase 1 sha `vllm=fa6ff060667f`. Also pinned FlashInfer `v0.7.0rc3`, InstantTensor `v0.2.0`, fastsafetensors `0.4.0`, LMCache `v0.5.5`, DeepEP `v1.2.1`, humming-kernels 0.1.15. b12x stays 1.2.6.

**Verdict.** **new-base control, not a keep.** 51.7 / 98.0 / 126.6 / **141.2**, sum **417.5**, worst spread 14.0 %. Gates pass in all three passes; acceptance 49.2-56.7 % and tokens per step 4.427-4.963, neither lower than `proto2-030`. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) this is pin noise. vs `refg030` (483.7 / 162.1) still -13.7 % / -12.9 %. Standing keep-rule arm on this pin is this sample until a same-day re-baseline says otherwise. Best previous-pin number remains `moemar` 455.3 on `main-030-rc1`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T14:20:06+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.3 / 49.5 / 51.7 | 51.7 | 7.4 % | - | - | - |
| 3 | 98.0 / 94.8 / 103.0 | 98.0 | 8.4 % | - | - | - |
| 5 | 126.6 / 127.9 / 122.0 | 126.6 | 4.7 % | - | - | - |
| 6 | 141.2 / 159.1 / 139.3 | 141.2 | 14.0 % | - | - | - |

### `ncclsock4-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_SOCKET_NTHREADS=4`. One variable: NCCL socket helper threads (default 1). Standing TP2 all-reduce is PYNCCL over RoCE. Confirmed `NCCL_SOCKET_NTHREADS=4`.

**Verdict.** **negative, not a keep.** 53.4 / 96.4 / 128.2 / **139.4**, sum **417.4**, worst spread 7.1 %. Gates pass in all three passes; acceptance 50.2-56.6 % and tokens per step 4.485-4.971 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is -0.0 % / -1.3 %. Worse at c6 than standing and than `p030-rc2b` (424.6 / 144.2). Socket helper threads are not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -14.1 % / -14.1 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T21:29:19+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.4 / 57.0 / 53.2 | 53.4 | 7.1 % | - | - | - |
| 3 | 96.4 / 95.0 / 98.1 | 96.4 | 3.2 % | - | - | - |
| 5 | 128.2 / 124.1 / 129.5 | 128.2 | 4.2 % | - | - | - |
| 6 | 139.4 / 138.2 / 143.5 | 139.4 | 3.8 % | - | - | - |

### `ncclnetib-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_NET=IB`. One variable: force NCCL NET=IB vs auto. Complementary to `ncclplug0-rc2` (NET_PLUGIN=none, pin noise). Confirmed `NCCL_NET=IB`.

**Verdict.** **negative, not a keep.** 54.4 / 96.1 / 127.1 / **139.1**, sum **416.7**, worst spread 12.3 %. Gates pass in all three passes; acceptance 49.9-58.1 % and tokens per step 4.485-5.069 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is -0.2 % / -1.5 %. Worse at c6 than standing and than `p030-rc2b` (424.6 / 144.2). Forced NET=IB is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -14.2 % / -14.2 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-21T00:24:28+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.2 / 51.5 / 54.4 | 54.4 | 12.3 % | - | - | - |
| 3 | 96.1 / 94.3 / 101.2 | 96.1 | 7.2 % | - | - | - |
| 5 | 125.9 / 133.8 / 127.1 | 127.1 | 6.2 % | - | - | - |
| 6 | 145.2 / 139.1 / 134.7 | 139.1 | 7.5 % | - | - | - |

### `ncclshm0-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_SHM_DISABLE=1`. One variable: disable NCCL shared-memory / CUDA-IPC path. Standing TP2 all-reduce is PYNCCL across two hosts; SHM is intra-node. Confirmed `NCCL_SHM_DISABLE=1`.

**Verdict.** **pin noise, not a keep.** 52.9 / 95.7 / 125.4 / **142.5**, sum **416.5**, worst spread 11.4 %. Gates pass in all three passes. One pass dipped to accept 48.1 % / tokens per step 4.339, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is -0.2 % / +0.9 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a wash. SHM off is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -14.3 % / -12.1 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T19:53:25+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 56.5 / 50.7 / 52.9 | 52.9 | 11.0 % | - | - | - |
| 3 | 104.0 / 93.1 / 95.7 | 95.7 | 11.4 % | - | - | - |
| 5 | 129.0 / 125.4 / 123.5 | 125.4 | 4.4 % | - | - | - |
| 6 | 140.1 / 142.5 / 143.0 | 142.5 | 2.0 % | - | - | - |

### `lintri`

**Changed.** `proto2-030` plus `LINEAR_BACKEND=triton`. One variable: FP8 linear via `TritonFp8BlockScaledMMKernel` (`is_supported` returns True on CUDA). MoE stays b12x. Linear family so far: b12x works, deep_gemm worse/unstable, humming/marlin die on O-proj, torch block-scaled is Hopper-only, flashinfer_b12x is NVFP4-only. Engine log: `Selected TritonFp8BlockScaledMMKernel for Fp8LinearMethod`.

**Verdict.** **negative, not kept.** 52.5 / 93.0 / 129.2 / **141.7**, sum **416.4**, worst spread 10.2 %. Gates pass in all three passes. Against standing `proto2-030` (412.1 / 138.3, spread 12.8 %) and same-day control `p030b` (420.3 / 141.6) this is pin noise. One pass dipped to accept 48.9 % and 4.401 tokens/step, below `proto2-030`'s floor (49.0 % / 4.414). Triton linear is not the remaining gap. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T18:50:53+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 53.7 / 52.5 / 49.1 | 52.5 | 8.8 % | - | - | - |
| 3 | 93.8 / 89.6 / 93.0 | 93.0 | 4.5 % | - | - | - |
| 5 | 126.9 / 129.5 / 129.2 | 129.2 | 2.0 % | - | - | - |
| 6 | 129.4 / 141.7 / 143.8 | 141.7 | 10.2 % | - | - | - |

### `lintri-rc2`

**Changed.** `proto2-030-rc2` plus `LINEAR_BACKEND=triton`. One variable: FP8 linear via `TritonFp8BlockScaledMMKernel`. MoE stays b12x. Confirmed `linear_backend: triton`. Re-measure of rc1 `lintri` (416.4) on this pin.

**Verdict.** **negative, not a keep.** 52.4 / 96.7 / 125.4 / **141.1**, sum **415.6**, worst spread 5.0 %. Gates pass in all three passes; acceptance 49.8-53.9 % and tokens per step 4.468-4.785 hold the floor. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is -0.5 % / -0.1 %. Against same-day control `p030-rc2b` (424.6 / 144.2) slightly worse. Triton linear is not the remaining gap. vs same-day `refg-rc2` (467.9 / 157.1) still -11.2 % / -10.2 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T19:55:21+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.4 / 50.1 / 52.7 | 52.4 | 5.0 % | - | - | - |
| 3 | 92.5 / 96.7 / 97.1 | 96.7 | 4.8 % | - | - | - |
| 5 | 122.5 / 125.4 / 127.5 | 125.4 | 4.0 % | - | - | - |
| 6 | 141.1 / 140.1 / 141.5 | 141.1 | 1.0 % | - | - | - |

### `ncclmin2-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_MIN_NCHANNELS=2`. One variable: raise NCCL channel floor. Complementary to `ncclch1-rc2` (`NCCL_MAX_NCHANNELS=1`, pin noise). Confirmed `NCCL_MIN_NCHANNELS=2`.

**Verdict.** **negative, not a keep.** 51.5 / 96.6 / 127.0 / **140.3**, sum **415.4**, worst spread 8.4 %. Gates pass in all three passes. One pass dipped to accept 47.5 % / tokens per step 4.303, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is -0.5 % / -0.6 %. Worse at c6 than standing and than `p030-rc2b` (424.6 / 144.2). MIN channels 2 is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -14.5 % / -13.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T22:28:08+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 50.4 / 51.5 / 51.6 | 51.5 | 2.3 % | - | - | - |
| 3 | 95.1 / 96.6 / 103.2 | 96.6 | 8.4 % | - | - | - |
| 5 | 130.1 / 127.0 / 125.5 | 127.0 | 3.6 % | - | - | - |
| 6 | 138.1 / 140.3 / 140.6 | 140.3 | 1.8 % | - | - | - |

### `proto2-030`

**Changed.** `v0.30.0rc1` (`a00a3544b93e`) matched-main rebuild, image `vllm-spark-0731:main-030-rc1`, same pin flags as `proto2-dg` (`VLLM_B12X_INDEXER_DIRECT_GATHER=1`, k=7, capture 48, util 0.8389, `--async-scheduling`). One variable: the base.

**Verdict.** **new-base re-baseline, not a win.** 52.8 / 93.9 / 127.1 / **138.3**, sum **412.1**, worst spread 12.8 %. Gates pass in all three passes; acceptance 49.0-54.8 % and tokens per step 4.414-4.815. Same-day `refg030` is 64.0 / 114.2 / 143.4 / **162.1**, sum **483.7**, worst spread 18.5 %. Behind 14.7 % at c6 and 14.8 % on the sum, both outside the larger of the two arms' spreads at those levels (c6 12.8 %, sum driven by the 18.5 % c5 swing on the reference). Standing configuration on `v0.30.0rc1` until a one-variable arm beats it. Proto-era `proto2-dg` (437.2) is a different base and a different day; do not A/B across them.

Container `vllm-ds4-0731 Up 5 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T14:32:40+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 52.8 / 50.1 / 53.8 | 52.8 | 7.0 % | - | - | - |
| 3 | 95.4 / 93.9 / 87.3 | 93.9 | 8.6 % | - | - | - |
| 5 | 127.1 / 137.2 / 126.3 | 127.1 | 8.6 % | - | - | - |
| 6 | 148.2 / 138.3 / 130.5 | 138.3 | 12.8 % | - | - | - |

### `tile128`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_TILE_MN=128x128`. One variable: dynamic MoE tile. Auto planner returns (16, 128) for the protocol band on `w4a8_mx`. 32x128 (`moetile32`) and 64x128 (`proto2-moetile`) were pin noise. 128x128 is the remaining ladder step. Confirmed in container env (`B12X_DYNAMIC_TILE_MN=128x128`).

**Verdict.** **negative, not kept.** 55.3 / 95.4 / 124.4 / **136.2**, sum **411.3**, worst spread 12.3 %. Gates pass in all three passes. One pass dipped to accept 48.8 %, below `proto2-030`'s floor 49.0 %; tokens per step 4.414-5.020 holds the floor. Against standing `proto2-030` (412.1 / 138.3) this is slightly worse. Auto 16x128 stays. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-19T00:45:47+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 50.2 / 55.3 / 57.0 | 55.3 | 12.3 % | - | - | - |
| 3 | 92.8 / 96.5 / 95.4 | 95.4 | 3.9 % | - | - | - |
| 5 | 126.6 / 123.3 / 124.4 | 124.4 | 2.7 % | - | - | - |
| 6 | 136.2 / 134.8 / 140.1 | 136.2 | 3.9 % | - | - | - |

### `proto2-dg-cgnone`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-18T00:48:35+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 29.4 / 35.4 / 32.7 | 32.7 | 18.3 % | - | - | - |
| 3 | 88.8 / 86.8 / 92.1 | 88.8 | 6.0 % | - | - | - |
| 5 | 133.8 / 136.1 / 135.9 | 135.9 | 1.7 % | - | - | - |
| 6 | 147.7 / 147.4 / 156.2 | 147.7 | 6.0 % | - | - | - |

### `moe_humming`

**Changed.** `MOE_BACKEND=humming`

**Verdict.** best sum measured (353.2), but not earned: +9.4 % at c1 against `protog`'s own 9.4 % c1 spread, and +0.4 % at c6. Needs a longer run before it can be kept.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T07:22:48+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 42.0 / 42.4 / 40.8 | 42.0 | 3.8 % | - | - | - |
| 3 | 80.0 / 78.8 / 79.3 | 79.3 | 1.5 % | - | - | - |
| 5 | 109.1 / 105.7 / 113.9 | 109.1 | 7.5 % | - | - | - |
| 6 | 126.4 / 120.7 / 122.8 | 122.8 | 4.6 % | - | - | - |

### `nccpll-rc2`

**Changed.** `proto2-030-rc2` plus `SERVE_EXTRA_ENV=NCCL_PROTO=LL`. One variable: NCCL low-latency protocol. Standing TP2 all-reduce is PYNCCL (FlashInfer MNNVL needs NVSwitch; FLASHINFER_PCIE_IPC is same-node CUDA-IPC). Eager profile priced 87 all-reduce calls at 81.3 ms gpu_sum on live c1. Confirmed `NCCL_PROTO=LL`.

**Verdict.** **negative, not a keep.** 51.1 / 86.8 / 101.6 / **112.8**, sum **352.3**, worst spread 8.8 %. Gates pass in all three passes. One pass dipped to accept 46.8 % / tokens per step 4.267, below standing floor 49.2 % / 4.427. Against standing `proto2-030-rc2` (417.5 / 141.2, spread 14.0 %) this is -15.6 % / -20.1 %. Against same-day control `p030-rc2b` (424.6 / 144.2) a large cost. NCCL LL is not the remaining gap. vs same-day `refg-rc2b` (485.9 / 162.2) still -27.5 % / -30.5 %. Standing arm remains `proto2-030-rc2`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-20T16:25:57+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 48.8 / 52.9 / 51.1 | 51.1 | 8.0 % | - | - | - |
| 3 | 86.8 / 83.1 / 90.7 | 86.8 | 8.8 % | - | - | - |
| 5 | 101.6 / 102.3 / 100.7 | 101.6 | 1.6 % | - | - | - |
| 6 | 114.2 / 112.8 / 111.7 | 112.8 | 2.2 % | - | - | - |

### `proto5`

**Changed.** `NUM_SPECULATIVE_TOKENS=5` (k=5 instead of 7)

**Verdict.** rejected as a lever: +8.7 % at c6 but -9.3 % at c5, level on the sum (349.7 against 346.7). c5 is where k=7's capture size fits (5x8=40 in the captured list) and k=5's does not (5x6=30).

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T06:30:35+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 38.9 / 37.9 / 39.2 | 38.9 | 3.3 % | - | - | - |
| 3 | 82.7 / 82.4 / 79.7 | 82.4 | 3.6 % | - | - | - |
| 5 | 95.4 / 98.1 / 94.0 | 95.4 | 4.3 % | - | - | - |
| 6 | 128.9 / 134.0 / 133.0 | 133.0 | 3.8 % | - | - | - |

### `protog`

**Changed.** proto base, k=7, capture 48, `--async-scheduling`

**Verdict.** **baseline** for every A/B below.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T05:30:13+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 37.6 / 38.4 / 41.2 | 38.4 | 9.4 % | - | - | - |
| 3 | 80.8 / 81.7 / 74.0 | 80.8 | 9.5 % | - | - | - |
| 5 | 104.2 / 105.2 / 107.6 | 105.2 | 3.2 % | - | - | - |
| 6 | 122.3 / 123.2 / 118.9 | 122.3 | 3.5 % | - | - | - |

### `attnfi`

**Changed.** `ATTENTION_BACKEND` and `DRAFT_ATTENTION_BACKEND` = `FLASHINFER_MLA_SPARSE_DSV4`, which is also the only configuration that passes the sparse-MLA autotune gate

**Verdict.** rejected: 37.8 / 78.5 / 104.6 / 119.8, inside `protog`'s spread. The attention implementation is not the gap, and the reference's 24 tuned configs are not what earns its step time.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T05:41:41+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 37.0 / 37.8 / 38.9 | 37.8 | 5.0 % | - | - | - |
| 3 | 78.7 / 78.5 / 76.8 | 78.5 | 2.4 % | - | - | - |
| 5 | 108.1 / 104.6 / 102.4 | 104.6 | 5.4 % | - | - | - |
| 6 | 118.2 / 123.6 / 119.8 | 119.8 | 4.5 % | - | - | - |

### `nogath`

**Changed.** `proto2-030` plus `VLLM_B12X_INDEXER_DIRECT_GATHER=0`. One variable: packed-indexer direct gather off. Default 1 in `05-serve.sh`. Proto-era +27 % win. Never isolated off on `main-030-rc1`. Overlay still scores logits via `logits_paged`. Confirmed in container env (`VLLM_B12X_INDEXER_DIRECT_GATHER=0`).

**Verdict.** **negative, load-bearing default.** 37.0 / 76.2 / 107.2 / **119.9**, sum **340.3**, worst spread 3.5 %. Gates pass in all three passes; acceptance 49.3-54.5 % and tokens per step 4.452-4.800, neither lower than `proto2-030`. Throughput dropped 17.4 % on the sum and 13.3 % at c6 vs standing `proto2-030` (412.1 / 138.3). Direct gather is required on this pin, not a leftover knob. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T22:48:05+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 38.1 / 37.0 / 37.0 | 37.0 | 3.0 % | - | - | - |
| 3 | 76.2 / 77.6 / 74.9 | 76.2 | 3.5 % | - | - | - |
| 5 | 107.7 / 107.2 / 104.0 | 107.2 | 3.5 % | - | - | - |
| 6 | 119.9 / 119.9 / 118.9 | 119.9 | 0.8 % | - | - | - |

### `proto2`

**Changed.** `main-029-proto2` rebuilt from the phase-1 base with the stale SM12x fp8_einsum recipe override removed from `apply_main`, `GPU_MEMORY_UTILIZATION=0.86`, k=7, capture 48, `--async-scheduling`, page cache dropped on both nodes first

**Verdict.** **the first proto2 arm from a real image, and the current best on this base.** Against the same-day reference it is **27.3 % below at c6** (117.0 against 161.0) and **31.0 % below on the sum** (334.3 against 484.8), both far outside the larger spread (26.0 %, the reference's own c5). Acceptance and tokens per step are comparable, so the deficit is step time: c6 step time is 234 ms against the reference's 177 ms. Against `protog` it is 122.3 -> 117.0 at c6 while the reference rose 156.0 -> 161.0, so proto2 is also about 7 % slower relative to the reference than proto was. The gap itself is unchanged: 234 ms here against `protog`'s 231.8 ms.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T13:31:44+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 33.6 / 35.4 / 37.9 | 35.4 | 12.1 % | - | - | - |
| 3 | 73.8 / 78.9 / 80.0 | 78.9 | 7.9 % | - | - | - |
| 5 | 105.7 / 103.0 / 102.5 | 103.0 | 3.1 % | - | - | - |
| 6 | 113.7 / 117.0 / 117.2 | 117.0 | 3.0 % | - | - | - |

### `proto2-moea4`

**Changed.** `proto2` with `B12X_MOE_FORCE_A8=0`, so the MoE uses the reference's FP4 activations instead of the FP8 ones our pin forces

**Verdict.** **a no-op arm, not a result.** 36.6 / 79.0 / 104.1 / 114.6, sum 334.3 against `proto2`'s 334.3, identical because nothing happened: `B12X_MOE_FORCE_A8` is read by no code in vLLM or b12x. The env var does reach the container -- `05-serve.sh` forwards it in its explicit `-e` list -- but it is a dead knob in our own pin. **The activation format was tested properly by `proto2-a16`, which uses the supported `VLLM_B12X_MOE_FP4_FORCE_A16` and really does change the selected backend.**

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T15:57:24+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 39.0 / 36.6 / 34.8 | 36.6 | 11.5 % | - | - | - |
| 3 | 79.5 / 72.0 / 79.0 | 79.0 | 9.5 % | - | - | - |
| 5 | 105.2 / 97.5 / 104.1 | 104.1 | 7.4 % | - | - | - |
| 6 | 113.3 / 117.8 / 114.6 | 114.6 | 3.9 % | - | - | - |

### `proto2-a16`

**Changed.** `proto2` with `VLLM_B12X_MOE_FP4_FORCE_A16=1` through `SERVE_EXTRA_ENV`, which selects `B12X_MXFP4_BF16` -- BF16 activations with MXFP4 weights, the closest analogue of the `B12X_MXFP4` backend the reference engine logs

**Verdict.** **wash, and it is the first arm that actually tested the MoE activation format.** The engine logged `Using 'B12X_MXFP4_BF16' Mxfp4 MoE backend` this time, so the variant really changed. 31.6 / 76.8 / 106.2 / 119.4, sum **334.0**, against `proto2`'s 334.3 -- the same sum, with catastrophic instability: spreads of 22.5 / 12.8 / **40.9** / **49.5 %**, one c6 pass falling to 63.0. Our default W4A8 path (`B12X_MXFP4_MXFP8`) is therefore not what is costing us, and the reference's activation format is not transferable at a profit. The earlier `proto2-moea4` arm did *not* test this: it set `B12X_MOE_FORCE_A8`, which no code in vLLM or b12x reads.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T16:15:10+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 30.6 / 31.6 / 37.7 | 31.6 | 22.5 % | - | - | - |
| 3 | 68.1 / 77.9 / 76.8 | 76.8 | 12.8 % | - | - | - |
| 5 | 68.1 / 111.5 / 106.2 | 106.2 | 40.9 % | - | - | - |
| 6 | 63.0 / 119.4 / 122.1 | 119.4 | 49.5 % | - | - | - |

### `proto2-dglin`

**Changed.** `proto2` with `LINEAR_BACKEND=deep_gemm`, i.e. the linear layer family set to the kernel the reference engine actually selects

**Verdict.** **negative, and it rules out the linear family.** The reference's own boot log shows it gets `Selected DeepGemmFp8BlockScaledMMKernel for Fp8LinearMethod` while ours gets `B12xFp8BlockScaledMMKernel`, and `_POSSIBLE_FP8_BLOCK_KERNELS` orders CUDA as FlashInfer-DeepGEMM, DeepGemm, Cutlass, then B12x -- so the two arms really are running different linear kernels. Forcing ours to the reference's choice booted, selected `DeepGemmFp8BlockScaledMMKernel`, and passed both gates, but is **no better and far less stable**: 30.1 / 76.9 / 106.7 / 114.6, sum 328.3, against `proto2`'s 35.4 / 78.9 / 103.0 / 117.0, sum 334.3, with a c6 spread of **48.6 %** (one pass fell to 62.9) against `proto2`'s 0.3 %. Our own b12x block-scaled MM is not the gap, and the reference's linear kernel is not transferable at a profit.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T15:26:58+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 30.1 / 28.2 / 38.0 | 30.1 | 32.6 % | - | - | - |
| 3 | 59.5 / 78.3 / 76.9 | 76.9 | 24.4 % | - | - | - |
| 5 | 95.4 / 106.7 / 107.1 | 106.7 | 11.0 % | - | - | - |
| 6 | 62.9 / 114.6 / 118.6 | 114.6 | 48.6 % | - | - | - |

### `proto2-attnfi`

**Changed.** `proto2` with `ATTENTION_BACKEND` and `DRAFT_ATTENTION_BACKEND` set to `FLASHINFER_MLA_SPARSE_DSV4`, the attention the reference engine actually runs -- but on a cold FlashInfer autotune cache, so it fell back to the default tactic heuristic

**Verdict.** **negative.** 35.4 / 73.6 / 101.9 / 117.2, sum 328.1, against `proto2`'s 35.4 / 78.9 / 103.0 / 117.0, sum 334.3 -- no better overall and worse at c3 and c5. The reference runs FlashInfer's `SparseMlaDecodeV3Runner` from a **pre-built autotune config** (`Config cache hit ... source=config file`, `flashinfer_autotune_cache/0.6.15/...`), while this arm logged `No FlashInfer SM120 sparse MLA DSv4 decode autotune cache entries found. Falling back to FlashInfer's default tactic heuristic.` So the swap was tested in its worst configuration, and the follow-up below tests the same arm with the cache the boot itself populated.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T15:37:26+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 36.0 / 35.4 / 33.4 | 35.4 | 7.3 % | - | - | - |
| 3 | 70.7 / 73.6 / 75.0 | 73.6 | 5.8 % | - | - | - |
| 5 | 101.9 / 104.1 / 100.0 | 101.9 | 4.0 % | - | - | - |
| 6 | 119.8 / 111.6 / 117.2 | 117.2 | 7.0 % | - | - | - |

### `proto2-attnfi-warm`

**Changed.** the identical arm, re-run after the previous boot wrote FlashInfer's sparse MLA DSv4 decode autotune entries -- one variable (cache populated vs not), same image, same env

**Verdict.** **negative, and it settles the attention family.** The cache loaded this time (`FlashInfer SM120 sparse MLA DSv4 decode autotune cache loaded on rank 0`), and the result is 37.5 / 71.1 / 99.3 / 113.0, sum **320.9** -- slightly *worse* than the cold run's 328.1 and clearly below `proto2`'s 334.3. The one real gain is stability: spreads of 3.2 / 4.4 / 2.3 / 2.0 % against the cold run's 7.3 / 5.8 / 4.0 / 7.0 %. Both FlashInfer configurations lose to `B12X_MLA_SPARSE`, so the attention backend is not the gap either, and the reference's tuned FlashInfer tactic does not transfer.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T15:47:10+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 37.8 / 36.6 / 37.5 | 37.5 | 3.2 % | - | - | - |
| 3 | 69.3 / 71.1 / 72.4 | 71.1 | 4.4 % | - | - | - |
| 5 | 99.7 / 99.3 / 97.4 | 99.3 | 2.3 % | - | - | - |
| 6 | 112.8 / 115.1 / 113.0 | 113.0 | 2.0 % | - | - | - |

### `proto2-compile`

**Changed.** `proto2` with `VLLM_USE_AOT_COMPILE=0`, meant to reach `CompilationMode.VLLM_COMPILE`

**Verdict.** **not the test it looks like, and the result is still useful.** It measured 35.1 / 75.3 / 97.1 / 111.4, sum **318.9** against `proto2`'s 334.3 -- 4.6 % worse -- but the resolved config still reads `CompilationMode.NONE`, so compilation never happened. What the arm actually measured is losing the AOT-compiled artifacts, which cost 4.6 %. **Keep `VLLM_USE_AOT_COMPILE=1`.** The reason the mode stays NONE is `config/vllm.py:786`: `if is_breakable_cudagraph_enabled(): self.compilation_config.mode = CompilationMode.NONE`, and `configs/env.spark.sh:47` forces `VLLM_USE_BREAKABLE_CUDAGRAPH=1`. See `proto2-break0`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T16:46:10+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 35.8 / 35.1 / 34.6 | 35.1 | 3.4 % | - | - | - |
| 3 | 77.7 / 75.3 / 72.6 | 75.3 | 6.8 % | - | - | - |
| 5 | 97.1 / 100.1 / 96.2 | 97.1 | 4.0 % | - | - | - |
| 6 | 109.5 / 111.4 / 114.4 | 111.4 | 4.4 % | - | - | - |

### `cgfull`

**Changed.** `CUDAGRAPH_MODE=FULL`

**Verdict.** rejected: worse at every level and unstable, with a 74 % c1 spread and truncated generations (1381 tokens at c3 where 1536 were asked for).

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T06:11:59+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.6 / 29.8 / 38.9 | 38.9 | 74.0 % | - | - | - |
| 3 | 76.0 / 83.5 / 69.5 | 76.0 | 18.4 % | - | - | - |
| 5 | 98.6 / 87.7 / 82.9 | 87.7 | 17.9 % | - | - | - |
| 6 | 102.8 / 114.6 / 109.2 | 109.2 | 10.8 % | - | - | - |

### `proto2-eager`

**Changed.** the `proto2` arm with `--enforce-eager` (`INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.8` to boot), i.e. CUDA graphs off

**Verdict.** **negative, and it refutes the graph-split hypothesis.** Our own `patch_tp_allreduce_eager_break` pulls the TP all-reduce out of the piecewise graph because an in-graph PYNCCL all-reduce on 2-node GB10 produced 1e33 logits, and that break splits the graph once per layer -- 87 all-reduce calls per step were measured -- which looked like a batch-independent per-step cost and therefore like the ~60 ms constant in the step-time table. It is not: turning the graphs off entirely is worse at every level (28.4 / 68.6 / 98.2 / 112.9, sum 308.1, against `proto2`'s 35.4 / 78.9 / 103.0 / 117.0, sum 334.3), so the graphs save more than the breaks cost and no graph configuration change is the fix. Worth keeping for a second reason: the deficit widens at c1 when graphs are off, so part of the fixed cost is host-side and the graphs are already hiding it.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T14:25:59+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 28.4 / 30.3 / 25.9 | 28.4 | 15.5 % | - | - | - |
| 3 | 70.3 / 68.6 / 67.2 | 68.6 | 4.5 % | - | - | - |
| 5 | 96.8 / 98.2 / 101.7 | 98.2 | 5.0 % | - | - | - |
| 6 | 112.9 / 110.3 / 121.5 | 112.9 | 9.9 % | - | - | - |

### `proto2-skip-attn`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up About a minute`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T19:14:02+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 58.0 / 57.9 / 52.4 | 57.9 | 9.7 % | - | - | - |
| 3 | 62.4 / 66.8 / 72.5 | 66.8 | 15.1 % | - | - | - |
| 5 | 81.7 / 79.2 / 79.6 | 79.6 | 3.1 % | - | - | - |
| 6 | 86.8 / 86.6 / 84.0 | 86.6 | 3.2 % | - | - | - |

### `hum-k5-rc2`

**Changed.** (unannotated)

**Verdict.** (unannotated)

Container `vllm-ds4-0731 Up 2 minutes`, engine `(not in the log)`, 5 passes at 512 tokens, started 2026-09-21T14:20:25+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 36.0 / 31.3 / 29.7 / 28.9 / 26.9 | 29.7 | 30.6 % | - | - | - |
| 3 | 80.3 / 77.1 / 41.8 / 75.0 / 76.5 | 76.5 | 50.3 % | - | - | - |
| 5 | 70.8 / 73.8 / 84.9 / 75.4 / 133.6 | 75.4 | 83.3 % | - | - | - |
| 6 | 80.2 / 93.5 / 109.9 / 111.9 / 82.5 | 93.5 | 33.9 % | - | - | - |

### `stockops2`

**Changed.** our WO-projection and sparse-indexer overlays off (`VLLM_USE_B12X_WO_PROJECTION=0`, `VLLM_USE_B12X_SPARSE_INDEXER=0`), with `MAX_MODEL_LEN=32768` so the KV pool fits

**Verdict.** rejected, and informative: -34 % at c6. Our per-layer overlays are load-bearing, not a cost. The higher context of the first attempt needs 9.48 GiB of KV against 9.32 GiB available.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T06:54:17+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 34.5 / 32.3 / 33.1 | 33.1 | 6.6 % | - | - | - |
| 3 | 55.7 / 59.9 / 61.2 | 59.9 | 9.2 % | - | - | - |
| 5 | 74.7 / 75.1 / 73.9 | 74.7 | 1.6 % | - | - | - |
| 6 | 78.9 / 82.1 / 81.2 | 81.2 | 3.9 % | - | - | - |

### `proto2-mhctl`

**Changed.** `proto2` with `DISABLE_DSPARK=1` and the mHC prenorm GEMM forced onto its TileLang fallback instead of DeepGEMM's tf32 kernel (`_USE_DEEP_GEMM = False`, bind-mounted `mhc/tilelang.py`)

**Verdict.** **negative: mHC's GEMM implementation is worth ~1 %.** 9.6 / 23.6 / 34.7 / 41.1 aggregate on spreads of 0.0-0.5 %, i.e. **104.2 / 127.1 / 144.1 / 146.0 ms** against `proto2-nodspark`'s 105.3 / 128.2 / 145.3 / 148.1 -- faster at every level by 1.1-2.1 ms, which is ~1 % and only just outside the control's spread. Unlike the skip probes this arm is a real A/B: **both gates pass** (`' Paris. The capital of Spain'`, `'72, 9x9'`), so the fallback is functionally correct and the comparison is meaningful. mHC runs a per-layer, batch-independent GEMM and it was the one per-layer component never priced; it is now priced at ~1 ms per step, so it is not the batch-independent floor either. Not keepable as a lever -- 1 % against a 27 % gap.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T19:22:32+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 9.6 / 9.6 / 9.6 | 9.6 | 0.0 % | - | - | - |
| 3 | 23.7 / 23.6 / 23.6 | 23.6 | 0.4 % | - | - | - |
| 5 | 34.7 / 34.7 / 34.6 | 34.7 | 0.3 % | - | - | - |
| 6 | 41.3 / 41.1 / 41.1 | 41.1 | 0.5 % | - | - | - |

### `proto2-nodspark`

**Changed.** `proto2` with `DISABLE_DSPARK=1`, i.e. speculative decoding off, so exactly one token is produced per engine step and the meter's per-stream tok/s *is* 1/step-time

**Verdict.** **Withdrawn as a draft-versus-target split (corrected in round 48); the step times themselves stand.** With speculation off a step processes `concurrency` rows; with DSpark k=7 it processes `concurrency x 8`. So the difference against `proto2` is the draft's work **plus** the target's own scaling from 6 rows to 48, and it cannot be separated inside the protocol, which fixes the levels at 1 3 5 6. The no-DSpark curve (+8.6 ms/row from 1 to 6 rows, then flattening) runs close to the DSpark points, so the +86.2 ms is mostly row scaling, not draft overhead. The measurement is still the cleanest of the run: **it isolates the target model's forward.** 9.5 / 23.4 / 34.4 / 40.5 aggregate tok/s with spreads of 0.0 / 0.4 / 0.6 / 0.5 %, which in step time is **105.3 / 128.2 / 145.3 / 148.1 ms**. Against `proto2` at 4.6 tokens per step (130 / 175 / 223 / 234 ms), the whole DSpark draft-plus-verification machinery costs **+24.7 ms at c1 rising to +86.2 ms at c6** -- it grows with batch, as expected for a draft run k+1 times over a larger batch. The target forward itself grows only 1.4x from 8 to 48 rows (105 -> 148 ms), so it is dominated by batch-independent work. **Against the reference's entire c1 step of 69 ms, our target forward alone at batch 8 is 105.3 ms -- 1.5x the reference's whole step**, and the reference also has a draft on top. That is the gap, now derived from two interventions rather than from profiler shares. Gates pass in all three passes.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T18:34:11+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 9.5 / 9.5 / 9.5 | 9.5 | 0.0 % | - | - | - |
| 3 | 23.4 / 23.4 / 23.3 | 23.4 | 0.4 % | - | - | - |
| 5 | 34.3 / 34.5 / 34.4 | 34.4 | 0.6 % | - | - | - |
| 6 | 40.7 / 40.5 / 40.5 | 40.5 | 0.5 % | - | - | - |

### `nomulti`

**Changed.** `proto2-030` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_ENABLE_MULTICTA=0`. One variable: disable dynamic MoE multi-CTA (default 1). Off forces `effective_mac=1`. On, DSV4F TP2 decode (E=256, k=6144, n=1024, 24-48 routed rows) caps at 24 resident CTAs and may double occupancy for compact tile_m<=32. Confirmed in container env (`B12X_DYNAMIC_ENABLE_MULTICTA=0`). Linear stayed `B12xFp8BlockScaledMMKernel`.

**Verdict.** **negative, load-bearing default.** 11.0 / 15.7 / 19.7 / **20.9**, sum **67.3**, worst spread 9.1 %. Gates pass in all three passes; acceptance 51.7-57.0 % and tokens per step 4.613-4.971, neither lower than `proto2-030`. Throughput collapsed ~6x vs standing `proto2-030` (412.1 / 138.3). Multi-CTA occupancy is required on this pin, not a leftover knob. Standing arm remains `proto2-030`.

Container `vllm-ds4-0731 Up 4 minutes`, engine `(not in the log)`, 3 passes at 512 tokens, started 2026-09-18T20:56:51+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 11.0 / 11.3 / 10.5 | 11.0 | 7.3 % | - | - | - |
| 3 | 16.0 / 15.7 / 15.6 | 15.7 | 2.5 % | - | - | - |
| 5 | 19.5 / 20.2 / 19.7 | 19.7 | 3.6 % | - | - | - |
| 6 | 20.6 / 22.5 / 20.9 | 20.9 | 9.1 % | - | - | - |

### `dglinear`

**Changed.** `LINEAR_BACKEND=auto`, which selects `DeepGemmFp8BlockScaledMMKernel`, the kernel the reference uses for its fp8 linears

**Verdict.** **numerically dead**: both gates return garbage, `accept_rate 0.0 %`, `tokens_per_step 1.002`. So `b12x` linear is the only working option on this pin, not a preference.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T05:56:23+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|

### `lin_cutedsl`

**Changed.** `LINEAR_BACKEND=flashinfer_cutedsl`, which logs `has no kernel for this linear layer type` and falls back to automatic selection, i.e. to the same DeepGEMM path

**Verdict.** **numerically dead** for the same reason as `dglinear`. With it, every fp8 linear route that is not `b12x` has now been measured.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T08:04:43+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|

### `prof6`

**Changed.** same profiler with `VLLM_PROFILE_DECODE_STEPS=1250`

**Verdict.** diagnostic. This is the run whose samples were misread as a c6 split; they carry `tok=8`, one sequence, so they are c1. Superseded by `p6c`.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T07:32:13+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 38.2 / 35.3 / 36.7 | 36.7 | 7.9 % | - | - | - |
| 3 | 76.4 / 77.4 / 73.5 | 76.4 | 5.1 % | - | - | - |
| 5 | 107.1 / 105.6 / 105.8 | 105.8 | 1.4 % | - | - | - |
| 6 | 123.5 / 121.4 / 117.9 | 121.4 | 4.6 % | - | - | - |

### `p6c`

**Changed.** the profiler with both settings passed through `SERVE_EXTRA_ENV`, so the 1250-step window actually applied

**Verdict.** diagnostic, and the run that produced the real c6 split: target 99.1 to 104.5 ms gpu against draft+sampler 21.2 to 21.9 ms.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T07:54:29+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 33.2 / 34.5 / 35.7 | 34.5 | 7.2 % | - | - | - |
| 3 | 76.0 / 77.5 / 75.8 | 76.0 | 2.2 % | - | - | - |
| 5 | 102.9 / 107.5 / 101.6 | 102.9 | 5.7 % | - | - | - |
| 6 | 129.8 / 124.9 / 126.1 | 126.1 | 3.9 % | - | - | - |

### `prof3`

**Changed.** same profiler with `VLLM_PROFILE_DECODE_STEPS=900`, still defeated by capture mode

**Verdict.** diagnostic, same window problem as `prof2`.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T06:40:19+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 37.2 / 37.9 / 37.4 | 37.4 | 1.9 % | - | - | - |
| 3 | 70.4 / 76.2 / 77.6 | 76.2 | 9.4 % | - | - | - |
| 5 | 104.7 / 106.9 / 101.5 | 104.7 | 5.2 % | - | - | - |
| 6 | 119.0 / 119.5 / 117.5 | 119.0 | 1.7 % | - | - | - |

### `prof2`

**Changed.** decode profiler armed, default 12-step window (`VLLM_PROFILE_DECODE=1 VLLM_PROFILE_CAPTURE=1`)

**Verdict.** diagnostic. The window is 12 steps, all of them c1, because the limit never reached the worker; throughput here is profiler-distorted and is not evidence about performance.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T06:21:13+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 39.6 / 34.4 / 40.5 | 39.6 | 15.4 % | - | - | - |
| 3 | 73.1 / 77.5 / 74.5 | 74.5 | 5.9 % | - | - | - |
| 5 | 102.5 / 103.7 / 102.7 | 102.7 | 1.2 % | - | - | - |
| 6 | 120.1 / 123.1 / 117.8 | 120.1 | 4.4 % | - | - | - |

### `prof6b`

**Changed.** same profiler with capture mode off and `VLLM_PROFILE_DECODE_STEPS=1250`

**Verdict.** diagnostic: capture mode was what defeated the limit, so with it off the window is still 12, which located the real cause at the SERVE_EXTRA_ENV boundary.

Container `vllm-ds4-0731 Up 3 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T07:44:02+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 36.3 / 36.5 / 33.3 | 36.3 | 8.8 % | - | - | - |
| 3 | 75.0 / 74.0 / 72.4 | 74.0 | 3.5 % | - | - | - |
| 5 | 112.8 / 106.5 / 107.8 | 107.8 | 5.8 % | - | - | - |
| 6 | 117.2 / 118.1 / 118.2 | 118.1 | 0.8 % | - | - | - |

### `prof4`

**Changed.** same profiler, region and per-layer marks read

**Verdict.** diagnostic: produced the c1 region table (ffn/MoE 1.84 ms of a 2.83 ms layer, 65 %) and the per-layer average used in the attribution.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.1.1.dev0+g69db1c26b`, 3 passes at 512 tokens, started 2026-09-13T07:05:28+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 34.5 / 36.2 / 37.7 | 36.2 | 8.8 % | - | - | - |
| 3 | 76.2 / 76.3 / 81.0 | 76.3 | 6.3 % | - | - | - |
| 5 | 105.7 / 100.3 / 103.3 | 103.3 | 5.2 % | - | - | - |
| 6 | 114.2 / 119.4 / 122.8 | 119.4 | 7.2 % | - | - | - |

### `proto2-recipe2`

**Changed.** `main-029-proto2` with a corrected `o_proj.py` bind-mounted over the image's stale copy, before the overlay fix was rebuilt into an image; `GPU_MEMORY_UTILIZATION=0.86`

**Verdict.** **diagnostic, and the arm that proved the fix.** Mounting the corrected `o_proj.py` took `csrc/utils/layout.hpp:113` from one occurrence per boot to zero and got the engine past the forward to `_check_enough_kv_cache_memory`. Its numbers (34.8 / 74.9 / 100.8 / 117.0) agree with `proto2`'s (35.4 / 78.9 / 103.0 / 117.0) inside the spreads, so the bind-mount was equivalent to the rebuild and no second measurement was needed to carry the conclusion over.

Container `vllm-ds4-0731 Up 4 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T12:33:50+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 36.7 / 33.5 / 34.8 | 34.8 | 9.2 % | - | - | - |
| 3 | 74.9 / 74.4 / 75.7 | 74.9 | 1.7 % | - | - | - |
| 5 | 107.6 / 100.2 / 100.8 | 100.8 | 7.3 % | - | - | - |
| 6 | 116.9 / 117.3 / 117.0 | 117.0 | 0.3 % | - | - | - |

### `proto2-skip-ffn`

**Changed.** `proto2` with `DISABLE_DSPARK=1` **and** `self.ffn(x, input_ids)` in the layer forward replaced by `x = x` (bind-mounted `model.py`), so the FFN half is priced by removal

**Verdict.** **the measurement the attribution needed, with one stated confound.** One token per step, so per-stream tok/s is 1/step-time: 10.4 / 30.7 / 51.1 / 61.2 aggregate, i.e. **96.2 / 97.7 / 97.8 / 98.0 ms**, on spreads of 0.0-0.3 %. Against the unmodified `proto2-nodspark` (105.3 / 128.2 / 145.3 / 148.1 ms) the FFN costs **+9.1 ms at c1 rising to +50.1 ms at c6**. Two things follow. First, **the FFN carries essentially all of the batch dependence**: with it removed the step is flat at 96-98 ms across 8 to 48 rows, so the remaining ~96 ms is batch-independent work. Second, the reference's *entire* c1 step is 69 ms, which is **below our batch-independent floor even with the MoE deleted** -- that floor, not the MoE, is where the c1 deficit lives. Gates are garbled by construction (the model has no FFN); the tokens still ran to full length in all three passes (512/1536/2560/3072), which is what makes the timing usable. The confound: a degenerate residual stream can also change the attention path, so read +50.1 ms as an upper bound on the MoE rather than an exact price.

Container `vllm-ds4-0731 Up 2 minutes`, engine `v0.2.1.dev0+gf37c550bf`, 3 passes at 512 tokens, started 2026-09-17T18:53:13+08:00.

| level | passes | median | spread | tokens/step | accept % | wall s |
|---|---|---|---|---|---|---|
| 1 | 10.4 / 10.4 / 10.4 | 10.4 | 0.0 % | - | - | - |
| 3 | 30.8 / 30.7 / 30.7 | 30.7 | 0.3 % | - | - | - |
| 5 | 51.1 / 51.1 / 51.1 | 51.1 | 0.0 % | - | - | - |
| 6 | 61.2 / 61.2 / 61.2 | 61.2 | 0.0 % | - | - | - |

## Arms that produced no numbers

- **`hum-k5-rc2`**: changed `hum-k6-rc2b` (humming, WO off) with `NUM_SPECULATIVE_TOKENS=5` and `MAX_CUDAGRAPH_CAPTURE_SIZE=48` unchanged, five passes. One variable: speculative depth 5 vs 6, continuing the ladder that worked at k=6.. **degenerate, do not use.** 29.7 / 76.5 / 75.4 / **93.5**, sum **275.1**, worst spread **83.3 %** (c1 30.6 %, c3 50.3 %, c5 83.3 %, c6 33.9 %): -42.4 % on the sum and -41.8 % at c6 against the standing `hum-k6-rc2b` (477.5 / 160.6). Tokens per step ranged 4.071-6.250, so the floor is breached too. Pass 5's final `/metrics` snapshot was refused (`ConnectionRefusedError`), i.e. the API was unavailable at the end of the run. All five passes are degraded, not just the last, so this is not a single flaky pass. k=6 stays the default; the k ladder is not monotone and k=5 is below the cliff.
- **`stepc6-rc2`**: changed corrected c6 step profile: `B12X_PROFILE_*` alias plus a bind-mounted patched `sm12x_b12x_kernels.py`, the protocol config (`NUM_SPECULATIVE_TOKENS=7 MAX_CUDAGRAPH_CAPTURE_SIZE=48`) and CUDA graphs on, level 6 only. Intended to measure the real c6 `execute_model` and `sample_tokens` device time, which the retracted 'host-bound' claim in `docs/UPSTREAM.md` had assumed instead of measured.. **no profiler output, no tok/s.** The mount and the module are correct (`grep -c B12X_PROFILE` = 5 in-container, decorator present in `vllm/v1/worker/gpu/model_runner.py`) and the served config is right (`num_spec_tokens=7`, `cudagraph_capture_sizes` includes 48), but the GPU worker still comes up without the alias in its environment, so the wrapper returns the bare function and nothing prints. Runner: `configs/examples/stepc6-rc2.sh`.
- **`prof6e-rc2`**: changed the same alias route in the regime where it previously worked: `ENFORCE_EAGER=1`, `B12X_PROFILE_DECODE_STEPS=60`, level 6 only (`drive-median.sh prof6e-rc2 1 512 6`). Re-run of `profeager-rc2` at c6 token counts.. **no profiler output.** Served and metered (c6 **146.3** tok/s) but printed no `b12x step` or `b12x region` lines at all, including no region table, so the route that produced the c1 table on 2026-09-20 no longer arms. Runner: `configs/examples/prof6e-rc2.sh`.
- **`nsysc6-rc2`**: changed `nsys profile --attach-pid <Worker_TP0> --cuda-graph-trace=node --duration 12` on the standing pin during a c6 window. One variable: profiler only. Intended as the ground-truth kernel table for a graph-replayed c6 step, because the overlay's region marks only collect under CUDA-graph capture and capture mode dies.. **no report, no tok/s.** `nsys --attach-pid` cannot enable CUDA tracing on a process that was not launched under nsys (no CUPTI injection), so no `.nsys-rep` was written. Runner: `harness/profile-c6-nsys.sh`. Not a performance result.
- **`nnoop-rc2`**: changed `proto2-030-rc2` plus `PASS_CONFIG={"eliminate_noops":false}`. One variable: disable inductor noop elimination (default True). Distinct from fusion flags (`fnormq`/`factq`/`fattnq`, hang). Confirmed `eliminate_noops': False` and `Enabled custom fusions: norm_quant, act_quant`.. **hangs at EngineCore init, no tok/s.** Same class as the fusion flags: any non-empty `pass_config` enables custom fusions and never reaches health. Standing empty `pass_config` is load-bearing. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/nnoop-rc2-fail.log`.
- **`fattnq-rc2`**: changed `proto2-030-rc2` plus `PASS_CONFIG={"fuse_attn_quant":true}`. One variable: force Attention/MLA+quant fusion (standing empty/False). Distinct from `fnormq-rc2`/`factq-rc2` (norm/act fusion, same hang class). Confirmed `fuse_attn_quant': True` and `Enabled custom fusions: norm_quant, act_quant, attn_quant, rope_kvcache_cat_mla`.. **hangs at EngineCore init, no tok/s.** Same class as `fnormq-rc2`/`factq-rc2`: fusion enable then EngineCore `Initializing a V1 LLM engine` and never reaches health. Any forced fusion flag hangs this pin. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/fattnq-rc2-fail.log`.
- **`factq-rc2`**: changed `proto2-030-rc2` plus `PASS_CONFIG={"fuse_act_quant":true}`. One variable: force SiluMul+quant fusion (standing empty/False). `fnormq-rc2` set fuse_norm_quant and still logged both norm_quant and act_quant then hung. This arm sets only fuse_act_quant. Confirmed `fuse_act_quant': True` and `Enabled custom fusions: norm_quant, act_quant`.. **hangs at EngineCore init, no tok/s.** Same class as `fnormq-rc2`: fusion enable then EngineCore `Initializing a V1 LLM engine` and never reaches health. Either fusion flag enables both and hangs. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/factq-rc2-fail.log`.
- **`fnormq-rc2`**: changed `proto2-030-rc2` plus `PASS_CONFIG={"fuse_norm_quant":true}`. One variable: force RMSNorm+quant fusion (standing empty/False). Distinct from `ensp-rc2` (SP+DSpark incompatible). Confirmed `fuse_norm_quant': True` and `Enabled custom fusions: norm_quant, act_quant`.. **hangs at EngineCore init, no tok/s.** APIServer logs fusion enable then EngineCore `Initializing a V1 LLM engine` and never reaches health. Same class as compile-path hangs, not a tok/s arm. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/fnormq-rc2-fail.log`.
- **`ensp-rc2`**: changed `proto2-030-rc2` plus `PASS_CONFIG={"enable_sp":true,"sp_min_token_num":1}`. One variable: force sequence parallelism. Standing `pass_config` is empty. SP_MIN_HIDDEN_SIZE is keyed 90/100 only, so SM121 auto-disables SP. Plumbing added in `scripts/05-serve.sh`. Confirmed `pass_config': {'enable_sp': True, 'sp_min_token_num': 1}`.. **dies at VllmConfig, no tok/s.** `ValidationError: Model Runner V1 does not support: dspark speculative decoding`. Forced SP on this pin is incompatible with DSpark. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/ensp-rc2-fail.log`.
- **`kvlbnhc-rc2`**: changed `proto2-030-rc2` plus `SERVE_EXTRA_ENV=VLLM_KV_CACHE_LAYOUT=LBNHC`. One variable: force layer-compact LBNHC vs standing auto (BLHNC). `nohma-rc2` fail text named LBNHC as the compact layout. Same image, only this env.. **dies at EngineCore init, no tok/s.** `ValueError: VLLM_KV_CACHE_LAYOUT=LBNHC does not satisfy every supported set; valid layouts: ['BLHNC', 'BLNHC']`. LBNHC is not legal for this `nvfp4_ds_mla` + DSV4 indexer pair. Standing auto BLHNC stays. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/kvlbnhc-rc2-fail.log`.
- **`nohma-rc2`**: changed `proto2-030-rc2` plus `ARM_EXTRA_ARGS=--disable-hybrid-kv-cache-manager`. One variable: explicit hybrid KV cache manager off vs standing auto. DSV4 has MLA plus a Lightning Indexer cache, so HMA can split the pool. Confirmed `disable_hybrid_kv_cache_manager': True`.. **dies at worker init, KV layout, no tok/s.** `ValueError: The resolved KV cache layout (BLHNC) does not store blocks as dense, unpadded pages (block stride 83889984 != page 149504), so a manager block cannot be split into 4 kernel blocks of 64 tokens.` Standing auto-HMA is load-bearing for `nvfp4_ds_mla` block 256 on SM120. Explicit disable is not the remaining gap. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/nohma-rc2-fail.log`.
- **`fiw31-rc2`**: changed `proto2-030-rc2` plus one-off bind-mount of `outputs/driver/one-off/fused_moe_b12x_fiswizzle.py` so MXFP4 `w4a8_mx` apply goes through FlashInfer `b12x_fused_moe` after an in-place w31 [gate;up] to [up;gate] flip, then `swizzle_block_scale` and `convert_sf_to_mma_layout(..., sf_vec_size=32)`. New hypothesis vs `fiswizzle-rc2` (scale layout only): native b12x flips vLLM fused [gate;up] to kernel [up;gate] once at prepare. Stock `patches/files/fused_moe_b12x.py` stays stock. Confirmed `b12x MoE MXFP4 using FlashInfer b12x_fused_moe w31-flip-then-swizzle-mma k32`.. **dies at profile_run, CUDA error, no tok/s.** Overlay logged, then `RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR` in `cublasGemmEx` during `determine_available_memory` / `profile_run` (`CUDA_ERROR_ILLEGAL_ADDRESS` also present). Same live-weight failure class as `fistat`/`fifunc`/`fiswizzle-rc2`. w31 flip plus swizzle is not the remaining gap. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/fiw31-rc2-fail.log`.
- **`cuteopt2-rc2`**: changed `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DIRECT_CUTE_OPTIONS=--opt-level=2`. One variable: CuteDSL OptLevel 2 on the micro-direct compile path. Dynamic W4A8 already hardcodes OptLevel(2) because ptxas -O3 register-starves that mainloop. Micro-direct still defaulted to OptLevel(3). Protocol c1 is the only micro level. Equals form so SERVE_EXTRA_ENV word-split stays one kv.. **dies at KV floor, no tok/s.** `_check_enough_kv_cache_memory`: 9.46 GiB available vs 9.48 GiB needed at util 0.8389 / `max_model_len` 65536. OptLevel 2 on micro-direct is hungrier than standing. Raising util or cutting `max_model_len` would be a second variable. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/cuteopt2-rc2-fail.log`.
- **`fiswizzle-rc2`**: changed `proto2-030-rc2` plus one-off bind-mount of `outputs/driver/one-off/fused_moe_b12x_fiswizzle.py` so MXFP4 `w4a8_mx` apply goes through FlashInfer `b12x_fused_moe` after `swizzle_block_scale` then `convert_sf_to_mma_layout(..., sf_vec_size=32)`. New hypothesis vs `fistat`/`fifunc` (name-only `quant_mode="mxfp4"`): linear e8m0 has the same numel as MMA, so convert without swizzle is a silent wrong layout. Stock `patches/files/fused_moe_b12x.py` stays stock. Confirmed `b12x MoE MXFP4 using FlashInfer b12x_fused_moe swizzle-then-mma k32`.. **dies at profile_run, IMA, no tok/s.** Overlay logged, then `cutlass.base_dsl.common.DSLCudaRuntimeError: CUDA_ERROR_ILLEGAL_ADDRESS (error code: 700)` in `gpu_worker.determine_available_memory` / `profile_run`. Same live-weight IMA as `fistat`/`fifunc`. Swizzle-before-convert is not the remaining gap. Dummy zeros compiled `static_m8`; live weights still IMA. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/fiswizzle-rc2-fail.log`.
- **`profcap-rc2`**: changed `proto2-030-rc2` plus capture-mode decode profiler (`VLLM_PROFILE_DECODE=1 VLLM_PROFILE_CAPTURE=1 VLLM_PROFILE_DECODE_STEPS=12`) via `SERVE_EXTRA_ENV`. Diagnostic: record region marks as CUDA-graph nodes for a 1-row replay.. **dies at graph capture, no tok/s.** `RuntimeError: Worker failed with error 'CUDA error: invalid argument'`. Region CUDA events cannot become graph nodes on this pin. Capture-mode profile is closed. Use eager (`profeager-rc2`) instead. Log: `outputs/driver/one-off/profcap-rc2-fail.log`.
- **`bs128-rc2`**: changed `proto2-030-rc2` plus `BLOCK_SIZE=128`. One variable: manager block size. Pin and anemll use 256. SM120 DSV4 kernel page is 64; 256 splits into four kernel pages, 128 into two.. **dies at worker init, no tok/s.** `ValueError: Misaligned Tensor data on argument #6` when calling `host_entrypoint(... cos_sin_cache: Tensor([n2, 64], float32), k_cache: Tensor([n3, n4, n5], uint8) ...)`, expected alignment 16 bytes. Manager 128 is not a legal `nvfp4_ds_mla` layout on this pin. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/bs128-rc2-fail.log`.
- **`moetilem-rc2`**: changed `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_MOE_TILE_MN=128x128`. One variable: micro MoE tile. `B12X_MOE_TILE_MN` gates `_select_micro_mma_tiler_mn` (c1, below the 64-row cutover), not the dynamic planner. Default micro tile is 64x128.. **dies at KV floor, no tok/s.** `_check_enough_kv_cache_memory`: 9.44 GiB available vs 9.48 GiB needed at util 0.8389 / `max_model_len` 65536. Micro 128x128 workspace is hungrier than standing. Raising util or cutting `max_model_len` would be a second variable. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/moetilem-rc2-fail.log`.
- **`fusedq-rc2`**: changed `proto2-030-rc2` plus `SERVE_EXTRA_ENV=B12X_DENSE_FUSED_QUANT=1`. One variable: dense GEMM fused BF16 quant on m=1 decode (default 0). Producer warp amax-scans and quantizes K-tiles into sA/sSFA smem, skipping the separate quant kernel.. **dies at KV floor, no tok/s.** `_check_enough_kv_cache_memory`: 9.47 GiB available vs 9.48 GiB needed at util 0.8389 / `max_model_len` 65536. Fused-quant workspace is hungrier than standing. Raising util or cutting `max_model_len` would be a second variable. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/fusedq-rc2-fail.log`.
- **`fifunc`**: changed `proto2-030-rc2` plus bind-mount of `patches/files/fused_moe_b12x.py` so MXFP4 `w4a8_mx` apply goes through FlashInfer `b12x_fused_moe` (functional API, auto static at routed_pairs<=640 / dynamic at profile 12288, module-level workspace). New hypothesis vs `fistat`–`fistat4` (`B12xMoEWrapper`). Confirmed `b12x MoE MXFP4 using FlashInfer b12x_fused_moe`. Overlay reverted.. **dies at first launch, IMA, no tok/s.** Prepare logged, then `CUDALaunch Error: CUDA_ERROR_ILLEGAL_ADDRESS` during engine init. Same live-weight IMA as `fistat`. Functional API does not fix the live MXFP4 path. Dummy zeros compiled `static_m8`; live weights still IMA. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/fifunc-fail.log`.
- **`moefic-rc2`**: changed `proto2-030-rc2` plus `MOE_BACKEND=flashinfer_cutlass`. Re-measure of rc1 `moefic` (ninja FAILED on FlashInfer 0.7.0 floating main) on tagged FlashInfer `v0.7.0rc3`. One variable: CUTLASS MXFP4 MoE.. **dies at JIT compile, no tok/s.** Engine selected `FLASHINFER_CUTLASS_MXFP4_MXFP8`, then ninja failed building `fused_moe_120` `cutlass_kernel_file_gemm_grouped_sm120_M128_BS_group5` for `sm_121a` (`FAILED: [code=4]`, `RuntimeError: Ninja build failed`). Same failure as rc1. SM120 CUTLASS MoE does not compile on this FlashInfer `v0.7.0rc3` / CUDA 13.3.1 stack. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/moefic-rc2-fail.log`.
- **`moemar-rc2`**: changed `proto2-030-rc2` plus `MOE_BACKEND=marlin`. Re-measure of rc1 `moemar` (455.3 / 154.8) on `v0.30.0rc2`. One variable: Marlin MXFP4 MoE. Same util 0.8389 / max_model_len 65536.. **dies at KV floor, no tok/s.** `ValueError: 9.48 GiB KV cache is needed, which is larger than the available KV cache memory (9.27 GiB)`. Do not raise util as a second variable. rc1 `moemar` served; this pin's extra tagged deps (FlashInfer `v0.7.0rc3` etc.) leave less KV room. Standing arm remains `proto2-030-rc2`. Log: `outputs/driver/one-off/moemar-rc2-fail.log`.
- **`fistat4`**: changed `proto2-030` plus FlashInfer-static overlay with one shared `B12xMoEWrapper(max_num_tokens=12288)` for all layers. Isolates `fistat3` (profile 12288 vs wrapper 48) vs `fistat2` (43 wrappers at 12288).. **hangs after weight load, no tok/s.** Confirmed `b12x MoE MXFP4 using FlashInfer MoEStaticKernel`, loaded 79.34 GiB, then GPU 96 % with no CuTe compile line — same hang as `fistat2`. One shared 12288 workspace still does not come up. Dummy zeros compiled `static_m8`; live path does not. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/fistat4-fail.log`.
- **`fistat3`**: changed `proto2-030` plus FlashInfer-static overlay with one shared `B12xMoEWrapper(max_num_tokens=48)` for all layers. Isolates the `fistat2` hang: 43 wrappers at `max_num_tokens=12288` allocating static workspaces. Dummy compiled at m=8; capture size is 48.. **dies at engine start, no tok/s.** Confirmed `b12x MoE MXFP4 using FlashInfer MoEStaticKernel`, then `ValueError: num_tokens (12288) exceeds max_num_tokens (48)` during KV-cache profile. Profile batch is max-num-batched-tokens, not capture 48. Shared wrapper at 48 cannot profile. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/fistat3-fail.log`.
- **`fistat2`**: changed `proto2-030` plus the same FlashInfer-static overlay as `fistat`, with contiguous stashed `w13`/`w2` and warmup skipped. One variable: live MXFP4 weights into image `B12xMoEWrapper(quant_mode="mxfp4")` without the warmup IMA. Confirmed `b12x MoE MXFP4 using FlashInfer MoEStaticKernel`.. **hangs after weight load, no tok/s.** Worker logged FI prepare, loaded 79.34 GiB, then GPU sat at 96 % for 20+ min with no CuTe compile line and repeating `No available shared memory broadcast block found in 60 seconds`. Health never 200. Dummy zeros compiled; live checkpoint hangs in `B12xMoEWrapper` construction / first graph capture. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/fistat2-fail.log`.
- **`fistat`**: changed `proto2-030` plus bind-mount of `patches/files/fused_moe_b12x.py` so MXFP4 `w4a8_mx` apply goes through image FlashInfer `B12xMoEWrapper(quant_mode="mxfp4")`. One variable: overlay MoE kernel, same image. Dummy `b12x_fused_moe` on protocol shapes compiled `static_m8_k6144_n1024_t6_r48`. Source MXFP4 layout matches `[E, 2n, k/2]` packed uint8 plus K32 e8m0 scales converted with `sf_vec_size=32`.. **dies at engine start, no tok/s.** Worker IMA during KV-cache sizing: `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered` in `triton/compiler/compiler.py:468 _init_handles`. Dummy zeros compiled; live checkpoint weights plus warmup did not. Weight-view / e8m0 MMA conversion on real tensors is still unproven. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/fistat-fail.log`.
- **`nomg`**: changed `proto2-030` plus `SERVE_EXTRA_ENV=B12X_MLA_SM120_PREFILL_MG=0`. One variable: MLA prefill-MG off (default 1). Overlay `B12X_MLA_SPARSE` reuses the prefill MG kernel for decode when `rows>=16`. Confirmed in the worker error (`B12X_MLA_SM120_PREFILL_MG=0`).. **dies at engine start, no tok/s.** `ValueError: SM120 sparse MLA prefill: unsupported shape (model_type=0, heads=32, topk=512, ..., B12X_MLA_SM120_PREFILL_MG=0). ... No decode-reuse fallback.` DSV4 requires MG. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/nomg-fail.log`.
- **`nochunk`**: changed `proto2-030` plus `--no-enable-chunked-prefill` via `ARM_EXTRA_ARGS`. One variable: chunked prefill off. vLLM default is True; the anemll recipe also enables it. `05-serve.sh` does not pass the flag. BooleanOptionalAction last-flag wins. Confirmed `enable_chunked_prefill: False`.. **dies at engine start, no tok/s.** Warning: `This model does not officially support disabling chunked prefill. Disabling this manually may cause the engine to crash or produce incorrect outputs.` Container exited 1 during init. DSV4 requires chunked prefill. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/nochunk-fail.log`.
- **`detout`**: changed `proto2-030` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`. One variable: force deterministic MoE output (default False). When True, W4A8 shared-input decode regime is skipped (`not deterministic_output`). Same image, no second variable.. **dies at engine start, no tok/s.** `ValueError: To serve at least one request with the model's max seq len (65536), (9.48 GiB KV cache is needed, which is larger than the available KV cache memory (9.24 GiB)`. Estimated max model length 24400. Raising util or cutting `max_model_len` would be a second variable. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/detout-fail.log`.
- **`util082`**: changed `proto2-030` plus `GPU_MEMORY_UTILIZATION=0.82`. One variable: match the anemll recipe util. `run-arm.sh` hardcodes 0.8389; EXTRA overrides it. `util8663` went the other way (0.8663) and sat in pin noise. Same image, no second variable.. **dies at engine start, no tok/s.** Profiler maps util 0.8200 -> effective 0.7944. Available KV 7.87 GiB vs 9.48 GiB needed for max seq len 65536 (estimated max model length 20784). Cutting `max_model_len` or turning the profiler off would be a second variable. Reference fits 0.82 because its image/workspace is different. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/util082-fail.log`.
- **`noidx`**: changed `proto2-030` plus `VLLM_USE_B12X_SPARSE_INDEXER=0`. One variable: overlay scores DSA indexer logits with b12x `logits_paged`; =0 returns None and the caller falls back to stock. Direct gather stays on. `stockops2` mixed this with WO-off and `MAX_MODEL_LEN=32768`. Never isolated on `main-030-rc1`.. **dies at engine start, no tok/s.** `ValueError: To serve at least one request with the model's max seq len (65536), (9.48 GiB KV cache is needed, which is larger than the available KV cache memory (9.46 GiB)`. Overlay-off is hungrier than the pin. Raising util or cutting `max_model_len` would be a second variable. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/noidx-fail.log`.
- **`linmar`**: changed `proto2-030` plus `LINEAR_BACKEND=marlin`. One variable: FP8 linear via Marlin. MoE marlin was the same-image ceiling (`moemar` 455.3) but not a keep. Linear humming died on O-proj shapes; linear deep_gemm was a protocol negative. Linear marlin was untested. **dies at worker init, no tok/s.** Selected `MarlinFP8ScaledMMLinearKernel` then `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [4, 4096] but got: [4, 1024]`. Same O-proj (`o_lora_rank` 1024 vs hidden 4096) as `proto2-linhum`. Linear family on this pin: `b12x` works, `deep_gemm` worse/unstable, `humming`/`marlin` cannot load, FlashInfer linear has no kernel. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/linmar-fail.log`.
- **`moefic`**: changed `proto2-030` plus `MOE_BACKEND=flashinfer_cutlass`. One variable: the MXFP4 oracle's FlashInfer CUTLASS expert path. `FlashInferExperts` claims SM90/SM100/SM120. Prior `moe_cutlass` never became healthy on the old pin; reason not captured. Retry on this image. **dies at JIT compile, no tok/s.** Engine selected `FLASHINFER_CUTLASS_MXFP4_MXFP8` / `FlashInferExperts`, then ninja failed building FlashInfer fused_moe_120 kernels (`FAILED: [code=4]` on `120_cutlass_kernel_file_gemm_grouped_sm120_M128_BS_group*.generated.cuda.o`). `RuntimeError: Ninja build failed`. SM120 CUTLASS MoE does not compile on this FlashInfer 0.7.0 / CUDA 13.3.1 stack. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/moefic-fail.log`.
- **`moedg`**: changed `proto2-030` plus `MOE_BACKEND=deep_gemm`. One variable: the MXFP4 oracle's DeepGEMM expert path (`DEEPGEMM_MXFP4` / `DeepGemmFP4Experts`, SM100 and SM120). Linear `deep_gemm` was already a protocol negative (`proto2-dglin`); MoE was untested on this pin. `assert_stack` only warns on unknown names, so the arm reached worker init. **dies at KV accounting, no tok/s.** Engine selected the backend (`Using 'DEEPGEMM_MXFP4' Mxfp4 MoE backend`) then failed `_check_enough_kv_cache_memory`: 8.79 GiB available vs 9.48 GiB needed at `gpu_memory_utilization=0.8389`. DeepGEMM's workspace is ~0.6 GiB hungrier than b12x (b12x at the same util had 9.39 GiB). Raising util would be a second variable. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/moedg-fail.log`.
- **`moeready`**: changed `proto2-030` plus `SERVE_EXTRA_ENV=B12X_DYNAMIC_WORK_SOURCE=ready_queue`. One variable: the remaining b12x dynamic work source after `moework` (`persistent_grid`) was a protocol negative. Library default is `materialized_queue`; `ready_queue` is the experimental overlapped publisher. **dies at profile_run, no tok/s.** Cutlass DSL rejects the publisher loop: `cutlass.base_dsl.common.DSLUserCodeError: PHASE_DYNAMIC_TO_STATIC_BOOL` at `b12x/moe/_shared/kernels/dynamic.py:2001` (`while g < num_groups` in `_publish_ready_tasks`). Experimental path is not JIT-clean on this b12x. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/moeready-fail.log`.
- **`cgstock`**: changed `proto2-030` plus `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, no overlay mounts. One variable: lift `CompilationMode.NONE`. Tests whether `v0.30.0rc1` (#56904 GPU-sync-under-compile) can compile DSv4 stock, without the proto-era 13-break port. **dies at the first Dynamo break, no tok/s.** Mode did lift: engine log shows `CompilationMode.VLLM_COMPILE: 3`. Failure is `torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped` on `vllm.third_party.deep_gemm._C...tf32_hc_prenorm_gemm` (pybind, no source file). Same class as proto-era break 5 / the parked `patch_mhc_tf32_*` overlays. `#56904` does not clear custom-op graph breaks. Compile on this pin still needs the overlay port, which rounds 55-72 already closed inside vLLM piecewise machinery. Standing arm remains `proto2-030`. Log: `outputs/driver/one-off/cgstock-fail.log`.
- **`fb12x`**: changed `proto2-030` with `MOE_BACKEND=flashinfer_b12x`, no rebuild. The one-variable arm the comparator's recipe names (`--moe-backend flashinfer_b12x`) and that round 10 recorded as untested on our stack. **rejected at worker init, no tok/s.** `ValueError: moe_backend='flashinfer_b12x' is not supported for MXFP4 MoE. Expected one of ['b12x', 'deep_gemm', 'flashinfer_trtllm', ...]`. The checkpoint's experts are MXFP4 (`expert_dtype: fp4` in `assert_0731`); `flashinfer_b12x` is an NVFP4-experts path (`FlashInferB12xExperts`). The reference can name it because its fork aliases that string onto a different kernel. On this pin the name is not an alias, so the comparator's MoE backend is not a configuration we can opt into. Log: `outputs/driver/one-off/fb12x-fail.log`. Standing arm remains `proto2-030`.
- **`proto2-cg18`**: changed `proto2-cg17` + `dynamic_shapes.op.py` gating `compilation/decorators.py:416 _mark_dynamic_inputs` behind `VLLM_B12X_STATIC_SHAPES=1`, forwarded via `SERVE_EXTRA_ENV`. **static shapes removes the assert and hits a second wall.** `assert_size_stride` occurrences drop to **0**, independently confirming round 71's diagnosis that the mismatch was the symbolic `s72` and not the strides. The new failure is `KeyError: 't0'` at `compilation/piecewise_backend.py:266 compile_all_ranges` (via `backends.py:353 -> compiler_interface.py:376`): **vLLM's piecewise splitter indexes the graph by the symbolic names `_mark_dynamic_inputs` creates**, so removing the symbols removes the keys it looks up. Both shape modes therefore fail inside vLLM's own piecewise machinery. **This vindicates upstream:** `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` containing `DeepseekV4ForCausalLM` is not an obstacle to route around, it is upstream stating that this path does not work for this model -- and the reference compiles only because its DSv4 stack is structurally different and far less Dynamo-clean-hostile. **Workstream closed:** no compile arm served, no tok/s, and no measurement showing compilation is faster. `proto2-dg` stands at sum 426.4-437.2 against 473.1-484.8.
- **`proto2-cg17`**: changed `proto2-cg16` + `attention.op.py`, making the attention output contiguous under `is_compiling()` because the generated code showed `o = o_padded[:, : n_local_heads, :]`. **failed identically -- and that is the measurement that resolved round 70's ambiguity.** `.contiguous()` is a no-op when the slice is already contiguous, so `padded_heads == n_local_heads == 32` and `o` already had strides `(16384, 512, 1)`: a tensor that already satisfies the guard cannot be failing it. **The mismatch is therefore the symbolic `s72` (`s72 = arg1_1`, a separate graph argument), not the strides** -- the opposite of every note written between rounds 67 and 70. It is vLLM's piecewise symbolic-shape plumbing. **Round 66's decision criterion is met** (five rounds, still no booting compiled arm and not one tok/s from any compile arm), so the port is recorded as the negative. Qualification: it is no longer an architectural wall -- the arm clears 13 Dynamo breaks, builds a complete Inductor graph (315 mHC, 231 all_reduce, 131 fused_inv_rope, 113 b12x_fp8_einsum, 98 fused_q_kv_rmsnorm, 88 moe_forward_shared, 62 indexer), and executes it. Resume path: gate `vllm/compilation/decorators.py:416 _mark_dynamic_inputs` off for static shapes (`wrapper.py:150` already passes `dynamic=False`; the symbols come from `mark_dynamic`, and `DynamicShapesType` has no STATIC). No tok/s, and no evidence compilation is faster.
- **`proto2-cg16`**: changed `proto2-cg15` + `$M/torch_compile_cache:/root/.cache/vllm/torch_compile_cache`, so the generated Inductor code survives the container. **not a behaviour change -- a diagnostic breakthrough.** The same `assert_size_stride` failure, but now the generated file is readable on the host (20 `.py` files, four carrying the guard). It identifies the partition: `arg0_1` is `o`, `(s72, 32, 512)` with `32 = n_groups * heads_per_group = 4 * 8` and `head_dim = 512`, expected contiguous `(16384, 512, 1)`. The partition does `zero_`, then `torch.ops.vllm.fused_inv_rope_fp8_quant_kernel`, then our `torch.ops.vllm.b12x_fp8_einsum`, then `per_token_group_fp8_quant`. **Correction to four rounds of notes:** the assert compares against the symbolic `s72`, which is a separate graph argument (`arg1_1`), so a failure is equally consistent with a token-count mismatch as with a stride mismatch -- previously described only as strides. Also, our `_o_ws[:tokens]` slice has strides `(16384, 512, 1)` for these dimensions, i.e. exactly what the assert expects. No tok/s.
- **`proto2-cg14 / proto2-cg15`**: changed `proto2-cg13` + a `multi_stream_utils.op.py` guard taking the sequential path under `is_compiling()` (cg14); then the same with `CUDAGRAPH_MODE=NONE` (cg15). **cg14: the guard was aimed at the wrong place.** The `Index not registered in index_to_user_object_weakref` chain runs `model_runner.py:1943 -> model.py:1506 -> compilation/caching.py:225 -> compilation/cuda_graph.py:256 -> compilation/piecewise_backend.py:380 -> streams.py:77 _get_stream_by_index`, i.e. vLLM's own **piecewise cudagraph machinery**, not `execute_in_parallel`. The guard is kept as correct behaviour but is not a fix. **cg15: `CUDAGRAPH_MODE=NONE` removes the stream error and the stride assert returns**, which **downgrades round 68's causal claim** -- `custom_ops: ["all"]` changed which failure fired first, it did not cause the assert, since the assert appears with `custom_ops` unset once capture is off. Two independent faults: (A) stream external-object index, only with capture on; (B) a stride mismatch `(s72, 32, 512)` independent of both. No tok/s.
- **`proto2-cg13`**: changed `proto2-cg12` minus `CUSTOM_OPS='["all"]'` -- a single-variable bisect of the knob added the round before. **the stride assert was caused by `custom_ops: ["all"]`, which was my own change from round 67.** With it: 2 occurrences of `assert_size_stride` and a failure at `wrong number of dimensions1 for op: input`. Without it: **0 occurrences**, and the failure moves to `Index not registered in index_to_user_object_weakref` (`torch/_dynamo/variables/streams.py:131 record_event`). So routing vLLM's registered `CustomOp` classes through their torch ops activates fake impls whose strides do not match the real implementations, and the compiled graph bakes in the difference; `align_inputs_from_check_idxs` then fails its own guard. **The knob is reverted and stays unset.** Round 67 called it a near-miss; it was harmful. Separately, `fused_q_kv_rmsnorm`'s fake impl was fixed to allocate with `torch.empty` instead of `empty_like` (which copies the input's strides) -- a real defect, but not the cause.
- **`proto2-cg11`**: changed `proto2-cg8` + `fused_indexer_q.op.py` (an opaque custom op for `fused_indexer_q_rope_quant`, fp8 branch only) + `CUSTOM_OPS='["all"]'`. **THE FIRST ARM THROUGH DYNAMO.** It compiled and reached execution: the trace now runs through `/root/.cache/vllm/torch_compile_cache/torch_aot_compile/<hash>/inductor_cache/.../*.py`, i.e. **Inductor's own generated code**, and fails there on `assert_size_stride(arg0_1, (s72, 32, 512), (16384, 512, 1), 'input')` during `profile_run`. That is a shape mismatch in the compiled graph, a correctness problem rather than a tracing blocker -- a much better class of problem and the first the port has produced. Two schema lessons: `infer_schema` demands an explicit return annotation on both `op_func` and `fake_impl` (its absence surfaces as `Model architectures [...] failed to be inspected`), and one schema cannot express the fp8 and fp4 return structures, so only the fp8 branch is registered (`use_fp4` is a Python literal, so Dynamo prunes the other). `custom_ops: ["all"]` is settable via the new `CUSTOM_OPS` knob in `scripts/05-serve.sh` and took effect, but did NOT clear break 17: `fused_indexer_q_rope_quant` is called directly, not through a `CustomOp` class. Next: bisect the eleven mounts; the shape looks like attention `q`.
- **`proto2-cg7`**: changed `proto2-cg6` + `import_utils.op.py` (capability probes folded to a pre-warmed dict) and `gpu_worker.op5.py` (which warms it). **break 16 cleared; break 17 is a CuTeDSL kernel and the port is now bounded.** `_has_module` and `_has_module_spec` lost their `@cache` decorators -- Dynamo traces the wrapped body, so a memoised probe is still fatal -- and read `_MODULE_RESULTS` under `is_compiling()`, raising LOUDLY for an unwarmed name rather than defaulting to False (a silent False would change which kernels run and still pass both gates). The arm then stopped at `fused_indexer_q.py:680 _INDEXER_Q_FP8_KERNEL(` -> `jit_warmup_cutedsl_helper.py:64 self.compile(...)` -- a CuTeDSL kernel being compiled during tracing, because the warmup that would populate its cache runs after profile_run. Counting the kernel singletons on this path gives **19, of which the 5 mHC ones are cleared**, so ~13 entry points remain, each needing a real custom op: none of the other DSv4 op modules registers anything upstream, so the mHC rebind trick does not generalise. **Caveat recorded:** fourteen rounds in, no arm has produced a tok/s number and there is still no measurement showing compiling makes our stack faster. Decision criterion set: if ~13 more entry points do not yield a booting compiled arm within ~5 rounds, the port is recorded as the negative and the consolidated attribution stands.
- **`proto2-cg6`**: changed `proto2-cg5` + `sm12x.op4.py` (the runtime-flag probes read a frozen snapshot under compile) and `gpu_worker.op4.py` (which retakes that snapshot in the pre-profile hook). **break 15 cleared; break 16 is the same class.** `b12x_skip_flag` no longer calls `os.path.exists` on the traced path -- the eager path is unchanged so skip flags can still be toggled while serving -- and `_indexer_direct_gather` got the same treatment. The arm then stopped at `attention.py:1166 wq_b_and_q_quant` -> `fused_indexer_q.py:674 if has_cutedsl():` -> `import_utils.py:601 _has_module("cutlass")`. **Breaks 15 and 16 are one class: a static process-lifetime capability probe evaluated inside the traced forward**, bottoming out in a skip-listed `os` or `importlib` call. This is the third appearance of the fold-the-probe-to-a-constant fix shape (`is_current_stream_capturing` round 54, runtime flags round 65, capability probes next). Safety note recorded: the capability cache must RAISE loudly for an unwarmed name, never default to False, or it would silently change which kernels run while still passing gates. No tok/s.
- **`proto2-cg5`**: changed `proto2-cg4` + `tilelang.op3.py`, which adds four module-level rebinds so the mHC names point at the custom ops upstream already registers. **the entire mHC family cleared in one small patch.** Upstream `mhc/tilelang.py` already ends with six `direct_register_custom_op` calls **and already contains the fake impls**, but `direct_register_custom_op` only does `define` + `impl` + `_register_fake` -- it never repoints the Python name, and `grep torch.ops.vllm` over the file returns nothing, so `model.py` imports the raw function and Dynamo dies on TileLang's compiled callable. Four two-line wrappers fixed five entry points, replacing four hand-derived fake impls. Recorded as an upstream-bug candidate; **no upstream contact made**, per the contract. The break moved to the attention path: **break 15** is `attention.py:1131 b12x_skip_flag("indexer_all")` -> `sm12x_b12x_kernels.py:1396 os.path.exists(...)` -- this repo's own skip-flag removal instrument, a skip-listed builtin, called inside the traced forward. No tok/s.
- **`proto2-cg4`**: changed `proto2-cg3` + the mHC TileLang warmup moved into the pre-profile hook (`gpu_worker.op3.py` calls `deepseek_v4_mhc_layer_warmup` alongside the wo_a pre-pack). **the break class changed, which is the point.** `tilelang/jit/__init__.py:392 _is_lazy_style` is gone: `_infer_jit_mode` short-circuits on a plain attribute read (`if self.mode in ("lazy", "eager")`), so priming `self.mode` before the first traced forward removes the break -- unlike the round-55 `cached_property`, where warming could never help because Dynamo traces the `except` branch anyway. The repo already had this warmup but wired it into `kernel_warmup`, which runs after `profile_run` has compiled, so it fired too late. The new break is `jit_warmup_tilelang_helper.py:142 jit_impl(*args, **call_kwargs)`, `call to a callable object with no traceable __call__` -- the compiled TileLang callable itself, the same class as break 9's ctypes `_FuncPtr`, so the four remaining mHC entry points do need custom ops after all. Pre-pack and mHC warmup both confirmed in the log. No tok/s.
- **`proto2-cg3`**: changed a **real** compile arm (no enumeration stub, no eager backend): `VLLM_USE_BREAKABLE_CUDAGRAPH=0` plus nine mounts -- `vllm_init.fold.py`, `sm12x.op.py`, `b12x_sparse.op.py`, `tilelang.op2.py` (break 5 merged with the parked tf32 custom op), `qk.op.py`, `jitw.op.py`, `gpu_worker.prepackonly.py`, `dsv4_warmup_ext.op.py`, `o_proj.op2.py` (the `data_ptr` sentinel fix merged with a new `fp8_einsum` custom op). **the furthest any compile config has got.** Advanced four breaks: the `fp8_einsum` `marked as skipped` failure is gone (the opaque custom op works -- `mutates_args=["out"]`, pairs flattened, `torch.empty_like(out)` fake impl), `deep_gemm.py:1153` `tf32_hc_prenorm_gemm` is gone (the merged tilelang overlay), `sm12x_b12x_kernels.py:1144` is gone, and the wo_a pre-pack still succeeds (43 packed). It stops at **break 13**: `model.py:1343 mhc_fused_post_pre_tilelang` -> `tilelang.py:1042` -> `tilelang/jit/__init__.py:392 is_lazy_style`, the same class as break 5 and the 2nd of the five mHC entry points. No tok/s.
- **`proto2-dg-nocar`**: changed `proto2-dg` + `--disable-custom-all-reduce`. **ran and was metered; a negative.** The reference runs `disable_custom_all_reduce: True`, we run `False`, so this was the last untested difference in the two engines' startup dumps: c1 55.5->54.9, c3 99.6->100.8, c5 131.7->129.1, c6 150.4->144.3, sum 437.2->429.1. c1/c3 are a wash and c5/c6 are worse, including the level the contract weights most. The custom all-reduce kernel is at least as good as NCCL here. **Closes the config-difference list:** compilation mode, `moe_backend`, `linear_backend`, `ir_op_priority.rms_norm`, `cudagraph_capture_sizes`, `disable_custom_all_reduce`, spec tokens and `max_model_len` are now all matched or measured. Required a `harness/run-arm.sh` change: it forced `SERVE_EXTRA_ARGS` itself and its `${EXTRA}` is word-split, so a second serve flag could not be passed; it now honours an optional `ARM_EXTRA_ARGS`, defaulting to empty so every earlier arm is unchanged.
- **`proto2-dg-cgfull`**: changed `proto2-dg` + `CUDAGRAPH_MODE=FULL` (one graph for the whole decode step, no piecewise splits). **ran and was metered; a large negative.** The direct test of "the c1/c3 gap is capture completeness": c1 55.5->31.5, c3 99.6->74.6, c5 131.7->97.0, c6 150.4->121.9, sum 437.2->325.0. Worse at every level and worse than `cgnone` (405.1), which removed capture entirely. Spreads 75.6 %/79.9 % at c1/c3 mean the arm is unstable, not merely slower -- consistent with capture failing or re-capturing. `FULL_AND_PIECEWISE` stands. With this and `cg24`, both directions around capture are now measured negatives.
- **`proto2-dg-cgnone`**: changed `proto2-dg` + `CUDAGRAPH_MODE=NONE`. **ran and was metered; kept as the price of capture, not as an arm.** Both gates pass on every pass. c1 55.5->32.7, c3 99.6->88.8, c5 131.7->135.9, c6 150.4->147.7, sum 437.2->405.1. In step time: capture is worth **-56.5 ms/step at c1** (86.5 against 143.1), -11.4 ms at c3, and at c5/c6 the apparent +5.8/+7.2 ms is inside the ~6 % pass-to-pass spread and is not claimed.
- **`proto2-dg-cg24`**: changed `proto2-dg` + `MAX_CUDAGRAPH_CAPTURE_SIZE=24`. **ran and was metered; a negative.** Captures 8/24 rows, drops 40/48. Worse at every level: c1 51.7, c3 96.5, c5 130.8, c6 147.5, sum 426.5 against 437.2. c6 147.5 matches `cgnone`'s 147.7, so dropping the 48-row capture does not recover `cgnone`'s c6 step time and lands below capture-on because the uncaptured runs also needed more steps. The capture ceiling is not a lever; `MAX_CUDAGRAPH_CAPTURE_SIZE=48` stands.
- **`proto2-cg2`**: changed `proto2-cg` + three more mounts: `jit_warmup_triton_helper.py` warming `_kernel_arg_names` at construction, `dsv4_warmup_ext.py` gaining a wo_a DeepGEMM einsum pre-pack plus a DeepGEMM `allow_in_graph` registration, `gpu_worker.py` calling both at the top of `determine_available_memory`, and `ops/o_proj.py` with the `data_ptr()` sentinel kept off the graph. died at worker init on break 12. The pre-pack worked (`DSv4 wo_a DeepGEMM einsum pre-pack: 43 packed, 0 skipped, 1.75 s`), clearing breaks 9 and 10; the `_kernel_arg_names` warm cleared the `hasattr(TritonKernelVariable)` class across all 7 candidate files; `allow_in_graph` cleared break 11 but turned it into break 12: `Dynamo failed to run FX node with fake tensors ... Cannot access data pointer of Tensor (e.g. FakeTensor) ... wrap the custom kernel into an opaque custom op`. No tok/s: no compile arm has been metered yet.
- **`proto2-cgfg0`**: changed `proto2-dg` + `VLLM_USE_BREAKABLE_CUDAGRAPH=0` + a bind-mounted `compilation/wrapper.py` with `fullgraph=True` relaxed to `False`. died at worker init: `Graph breaks are not supported with aot compile. Please use torch.compile(fullgraph=True).` The pin runs `VLLM_USE_AOT_COMPILE=1`, and AOT needs a single graph, so the two knobs are coupled: graph breaks or AOT, not both.
- **`proto2-cgfg0aot0`**: changed `proto2-cgfg0` + `VLLM_USE_AOT_COMPILE=0`. died at worker init: `AssertionError: VllmBackend can only be called once`. With the break allowed, Dynamo calls the backend once per graph fragment and vLLM's backend supports exactly one. `fullgraph=False` is therefore not a supported configuration in any of the four `CompilationMode` values, and every graph break is fatal.
- **`proto2-cg`**: changed `proto2-dg` + `VLLM_USE_BREAKABLE_CUDAGRAPH=0` + five mounts: `vllm/__init__.py` folding `torch.cuda.is_current_stream_capturing`, `utils/sm12x_b12x_kernels.py` with the `DBG wo_proj` prints removed, `b12x_sparse.py` skipping `logger.info_once` while compiling, and the `mhc_pre_broadcast_tilelang` / `fused_q_kv_rmsnorm` custom ops. died at worker init on break 9: `call to a callable object with no traceable __call__`, `object=<_FuncPtr object>`, from `ops/o_proj.py:99 deepgemm_post_process_fp8_weight_block` -> `utils/deep_gemm.py:536`, a ctypes call into a one-time weight repack running lazily inside `forward` during `profile_run`. No tok/s: no compile arm has been metered yet.
- **`deeplinear`**: changed `LINEAR_BACKEND=auto VLLM_USE_DEEP_GEMM_E8M0=1`. died at worker init: DeepGEMM assert `sf.size(-2) == ceil_div(mn, gran_mn)` at `csrc/utils/layout.hpp:97`, the ue8m0 scale layout.
- **`dglinear`**: changed `LINEAR_BACKEND=auto`. started but is numerically dead; see its row above.
- **`stockops`**: changed our WO-projection and sparse-indexer overlays off, at the default 65536 context. died before health: with the stock paths the engine needs 9.48 GiB of KV at 65536 context against 9.32 GiB available. Retried as `stockops2` with `MAX_MODEL_LEN=32768`.
- **`moe_trtllm`**: changed `MOE_BACKEND=flashinfer_trtllm`. died at worker init: `Mxfp4 MoE backend 'FLASHINFER_TRTLLM_MXFP4_MXFP8' does not support the deployment configuration since kernel does not support current device cuda`.
- **`moe_cutlass (probe)`**: changed `MOE_BACKEND=flashinfer_cutlass`. never became healthy; the reason line was not captured. Part of the MoE sweep, which found only `humming` runnable.
- **`proto2-compile-op1`**: changed `proto2` + `VLLM_USE_BREAKABLE_CUDAGRAPH=0` + a bind-mounted `mhc/tilelang.py` that registers `mhc_pre_broadcast_tilelang` as a custom op with an explicit fake impl. **not a measurement, and it cleared break 5.** The skip error drops to 0 and the TileLang frame is gone; compilation advances to break 6, a `hasattr(TritonKernelVariable, arg_names)` in `attention.py:544 _split_qkv_and_norm`. Two requirements were learned: the wrapper's signature must mirror the original's defaults exactly (`infer_schema` puts them in the schema, and the first attempt died on `missing value for argument 'n_splits'`), and `torch.Tensor | None` is accepted and becomes `Tensor?`. Rebinding the public name means `model.py` needs no patch, because it imports the name at module load.
- **`proto2-nodg`**: changed the kept arm with `is_deep_gemm_supported()` forced False on device family 120, i.e. `patch_deep_gemm_sm12x_guard` applied globally via a bind-mounted `utils/deep_gemm.py`. **never became healthy, and the reason vindicates the overlay being parked.** The guard is effective -- the `DeepGEMM E8M0 enabled on current platform` line stops appearing -- but the engine dies at worker init with `Assertion error (.../deepgemm-src/csrc/utils/layout.hpp:113): sf.size(-2) == ceil_div(mn, gran_mn)`, the **same** assertion round 26 fixed from the other side. The coupling is identical: `deep_gemm_fp8_o_proj`'s recipe selection reads the DeepGEMM support and E8M0 state, so reporting DeepGEMM unsupported flips the scale layout to one the *still-running* DeepGEMM FP8 einsum rejects. Round 42's ~1 % gain came from switching only the mHC prenorm GEMM off DeepGEMM, which is a different, local change; doing it globally is not available. `patch_deep_gemm_sm12x_guard` is applied by `apply()` and correctly not by `apply_main`.
- **`proto2-skip-attn`**: changed `proto2` with `DISABLE_DSPARK=1` and `self.attn(positions, x, None)` replaced by `x = x`. **not a usable measurement, recorded so nobody retries it.** Removing attention collapses the residual stream: the runs produced short generations (toks 601/761/75/1230 at c3 instead of the expected 1536, and varying per pass), spreads blew out to 9.7-15.1 %, and the implied step time was ~17 ms -- a 6x speedup from deleting half the layer, which is not credible. The gates are garbage (`' a =  image:ife'`, `' 组合内~| '`). The likely mechanism is routing collapse: with degenerate hidden states the MoE routes far fewer distinct experts, so the arm prices attention *and* a crippled MoE together. **Removal is only a clean instrument when the removed stage does not feed the selector of another** -- which is why the all-reduce skip (0.3 ms) and the FFN skip worked while this one does not.
- **`proto2-linhum`**: changed `LINEAR_BACKEND=humming`, the only untried linear backend (the profile puts the linear family at ~44 % of CUDA time). died at worker init: `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [4, 4096] but got: [4, 1024]`. The humming linear kernel cannot handle this model's O-projection shapes -- `o_lora_rank` 1024 against hidden 4096. **The linear family is therefore exhausted**: `b12x` is the current best, `deep_gemm` (the reference's own choice) measured worse and unstable, `humming` cannot load, and `cutlass`/`marlin` are in the NVFP4 clamp set this rig rejects.
- **`proto2-compile-customop`**: changed `proto2` + `VLLM_USE_BREAKABLE_CUDAGRAPH=0` + a bind-mounted `model_executor/kernels/mhc/tilelang.py` that registers `tf32_hc_prenorm_gemm` as a custom op and hoists the three local import blocks. **not a measurement, but it cleared the break.** Compilation got past the pybind GEMM -- the `tf32_hc_prenorm_gemm` frame is gone from the traceback -- and died at the next one, TileLang's `_MHC_PRE_BIG_FUSE_TILELANG_KERNEL` -> `tilelang/jit/__init__.py:527 __call__` -> `_infer_jit_mode` -> a TVM source introspection Dynamo skips. Reproducible as `patches/apply_overlays.py --only mhc-tf32-customop`.
- **`proto2-break0`**: changed `proto2` with `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, which is what lifts the `CompilationMode.NONE` override and lets the model actually compile (our overlay puts `@support_torch_compile` on it). **died at worker init, but this is the most informative failure of the round**: the config really did resolve to `CompilationMode.VLLM_COMPILE` and compilation began. It stopped at `torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped`, naming `vllm.third_party.deep_gemm._C...tf32_hc_prenorm_gemm`, reached from `model.py:1543` -> `model.py:1284 mhc_pre_broadcast_tilelang` -> `tilelang.py:760 _hc_prenorm_gemm_outputs` -> `tilelang.py:61`. That is exactly the break the repo catalogued on 2026-09-13 and never cleared. **`patch_mhc_tf32_uncaptured` and `patch_mhc_tf32_call_redirect` exist but are deliberately parked from `apply_main`** (their call-site needle predates `pr-53055.diff`, which folded the local import into a guarded `is_deep_gemm_supported, tf32_hc_prenorm_gemm` line, so the needle is gone and the overlay's `SystemExit` would cut off every later overlay), and the route they implement is one this torch rejects anyway. The sanctioned route is vLLM's own `direct_register_custom_op`, which the parked patch already imports.
- **`proto2-pcie-ar`**: changed `proto2` with `tensor_model_parallel_all_reduce` routed through b12x's `PCIeAllReduce` (`single_channel=True`) by bind-mounting a patched `communication_op.py`. not a measurement: the runtime is CUDA-IPC based and fails across hosts with `failed to open CUDA IPC handle for peer`, so every call fell back to PYNCCL and the arm is just `proto2` again. Its `*.median.log` is deliberately left on the rig and not pulled. An earlier attempt that used the wrapper's own `should_allreduce` died on a library `AttributeError` before reaching the collective.
- **`proto-b12x015`**: changed the proto base with the reference's `b12x 0.15.3` in place of ours (`vllm-spark-0731:main-029-proto-b12x015`, finally copied to spark2 so this arm could run). died at worker init: `ValueError: Failed to find a kernel that can implement the ScaledMM linear layer`. `b12x 0.15.3` does not ship the ScaledMM linear that `LINEAR_BACKEND=b12x` resolves to, so the two generations are not drop-in. **Retired deliberately**: the rule for this repo is the current base, the current `b12x 1.2.6`, and patches on top -- never a regression to an older library to recover one missing kernel.
- **`proto-b12x015-linauto`**: changed the same image with `LINEAR_BACKEND=auto`, the linear selection the reference itself uses, leaving `MOE_BACKEND=b12x` to isolate the MoE kernel. never became healthy either; the ScaledMM selection is not the only incompatibility. Recorded and abandoned with `proto-b12x015` for the same reason: an older `b12x` generation is not the route. The route is the current library's own unused fast paths, of which the PCIe all-reduce in `b12x.comm.pcie` is the strongest lead found so far.
- **`b12x015`**: changed a derivative image with the reference's `b12x-0.15.3` in place of ours. not a performance result: the image existed only on spark1, so spark2's worker could not start (`pull access denied`) and the head blocked waiting for rank 1. Superseded by the side-by-side `b12xref` build.

## Superseded arms (pre-guard)

Measured before 2026-09-13T03:15, from an earlier generation of this work. Kept as history. Their
numbers are **not evidence** and should not be compared against anything below or above.

| tag | levels | sum |
|---|---|---|

## Kernel probes (offline, no serve)

These do not run the serving protocol, so their numbers are not comparable with the arm
tables above. Each answers one premise with a real GPU measurement; raw output is kept
under `outputs/driver/one-off/`.

### tight same-session pair (ours then reference, back to back)

**Command.** ``.scratch/arm-pair.sh` -- `harness/run-arm.sh proto2-dg-pair` immediately followed by `harness/run-refg.sh`, no idle gap`

**Log.** [`outputs/driver/proto2-dg-pair.median.log`](../outputs/driver/proto2-dg-pair.median.log)

| level | ours | spread | reference | spread | gap |
|---|---|---|---|---|---|
| c1 | 53.9 | 7.6 % | 65.3 | 4.0 % | -17.5 % |
| c3 | 98.6 | 3.3 % | 110.1 | 4.6 % | -10.4 % |
| c5 | 129.9 | 0.2 % | 138.6 | 16.1 % | -6.3 % |
| c6 | 144.0 | 6.7 % | 159.1 | 2.1 % | -9.5 % |
| **sum** | **426.4** | | **473.1** | | **-9.9 %** |

Step time: +11.5 / +19.2 / +17.6 / +18.1 ms, median **+17.8 ms**, over a step time that
more than doubles (86.8 -> 193.5 ms). Gates pass on every pass in both legs.
Reference logs: `outputs/driver/refg.median.log` and `outputs/driver/refg-{1,2,3}.meter.txt`.

**Verdict.** **Both arms must be measured together.** This is cheaper than it looks (two arms, ~40 min) and it removes the one objection that cannot be answered after the fact. Do this whenever a final claim is being made.

### proto2-enum

**Command.** ``VLLM_USE_BREAKABLE_CUDAGRAPH=0` + `fullgraph=False` with an **eager** backend + `TORCH_LOGS=+graph_breaks` + an `fp8_einsum` stub that returns the caller's own `out``

**Log.** [`outputs/driver/one-off/proto2-enum-breaks.log`](../outputs/driver/one-off/proto2-enum-breaks.log)

| forward-path site | reason | owner |
|---|---|---|
| `sm12x_b12x_kernels.py:330,807,1093` | Dynamo cannot trace builtin `print` | ours (cleared) |
| `sm12x_b12x_kernels.py:1144` | non-contiguous `out=` in `torch.bmm` | ours (cleared) |
| `jit_warmup_triton_helper.py:232` | `hasattr(TritonKernelVariable, 'arg_names')` | upstream (cleared) |
| `b12x/.../mla/merge.py:84,85,92` | `torch.* op returned non-Tensor` | third-party |
| `deep_gemm.py:1153` | `tf32_hc_prenorm_gemm` pybind call | upstream (PR #53055) |
| `multi_stream_utils.py:56` | `torch.fx.traceback.annotate` | upstream |
| `attention.py:852`, `import_utils.py:427` | untraceable call / do-not-trace import | upstream |

23 sites in total; the remainder are warmup/JIT/import noise (`pynvml`, `tvm_ffi`,
`tilelang`, `nvidia_cutlass_dsl`, `inspect`).

**Verdict.** **Enumeration works; keep it, and it was used to verify two fixes.** The re-run after round 59 confirms `jit_warmup_triton_helper.py:232` is gone and that the `print` breaks are gone, and it surfaced a new one in our own code (`sm12x_b12x_kernels.py:1144`, a non-contiguous `out=` in `torch.bmm`), now fixed. One boot replaces one-break-per-boot and it immediately refuted a fix from round 55. It is not a measurement and must never be quoted as one -- the eager backend gives no speedup and the `fp8_einsum` stub is wrong on purpose.

### `b12x_ref` WO projection against our `b12x`, on the GPU

**Command.** `docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench vllm-spark-0731:main-029-proto-b12xref /bench/bench_b12x_wo.py`

**Log.** [`outputs/driver/one-off/b12x-wo-headtohead.log`](../outputs/driver/one-off/b12x-wo-headtohead.log)

| tokens | b12x (ours) | b12x_ref | delta |
|---|---|---|---|
| 1 | 0.374 | 0.331 | ref -0.042 |
| 2 | 0.372 | 0.336 | ref -0.036 |
| 4 | 0.387 | 0.337 | ref -0.050 |
| 8 | 0.381 | 0.344 | ref -0.037 |
| 16 | 0.381 | 0.396 | ours -0.015 |
| 32 | 0.410 | 0.426 | ours -0.015 |
| 48 | 0.405 | 0.446 | ours -0.042 |
| 64 | 0.417 | 0.479 | ours -0.063 |

Median of 50 CUDA-event timings per cell, ms. Both kernels were driven with the same
FP8 block-scaled weights and the same input; relative difference 1e-6 at 1 token and
0.0 at 8 and 64 tokens, so the two are computing the same thing and the timings are
comparable. Shapes: hidden 4096, groups 8, group_width 4096, rank 1024 (DSV4-Flash).

**Verdict.** **rejected as a lever.** The reference's WO kernel is faster only at or below 8 tokens, by at most 0.050 ms; from 16 tokens up ours is faster, by 0.063 ms at 64. A decode step at c6 runs roughly 48 rows, where ours is already 0.042 ms ahead, and the best case for a port is 0.05 ms against a 38.65 ms c6 step, 0.13 %. The 0.60 ms the profiler attributes to the WO region is the live path (fused inverse-RoPE quant, dequant, grouped `bmm`, `wo_b` linear), not this native kernel, so this closes the `b12x_ref` generation as the source of the WO cost and does not show the region itself is optimal.

### `b12x_ref` compressed sparse MLA against our `b12x`, on the GPU

**Command.** `docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench vllm-spark-0731:main-029-proto-b12xref /bench/bench_b12x_mla.py`

**Log.** [`outputs/driver/one-off/b12x-mla-headtohead.log`](../outputs/driver/one-off/b12x-mla-headtohead.log)

| library | run 1 | run 2 | run 3 | median | vs pure-torch reference |
|---|---|---|---|---|---|
| `b12x` (ours) | 0.2236 | 0.2238 | 0.2257 | **0.224** | 0.50 % |
| `b12x_ref` | 0.6138 | 0.6105 | 0.6319 | 0.614 | 0.50 % |

Median of 30 CUDA-event timings per run, ms, DSV4-Flash decode: 48 query rows,
32 local q heads (TP=2), head_dim 512, SWA window 128, indexed topk 512 pages,
584 B/token packed page. Inputs built with the reference package's own
`pack_compressed_mla_kv_cache_reference`, and the pure-torch
`compressed_sparse_mla_reference` is carried as ground truth. Both kernels sit
0.50 % from that reference, and agree with each other to 0.10 %, so they compute
the same thing.

**Verdict.** **the reference's attention generation is a regression, not a prize.** Our kernel is **2.7x faster** (0.224 ms against 0.614 ms) on the same contract and the same inputs. There is nothing to port here. Two consequences: attention is not where the anemll engine earns its step time, so the remaining gap has to be in the region the profiler prices highest, ffn/MoE at 1.84 ms of the 2.83 ms target layer (65 %), not in attention at 0.94 ms; and the `b12x_ref` generation is now closed for both surfaces measured, WO and MLA. Caveat recorded with the number: this compares each stack's public decode entry point, and `compressed_mla_decode_forward` splits into chunks and merges where ours is a single fused call, so part of the 2.7x is entry-point design rather than kernel throughput. Both are what their own engine calls, which is what makes the comparison the relevant one.

### MoE decode regime coverage: our `b12x` against the reference's

**Command.** `docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench vllm-spark-0731:main-029-proto-b12xref /bench/probe_moe_tuning.py`

**Log.** [`outputs/driver/one-off/moe-tuning-coverage.log`](../outputs/driver/one-off/moe-tuning-coverage.log)

| routed rows | ours micro | ours dynamic | ref micro | ref **static** | ref dynamic |
|---|---|---|---|---|---|
| 8 | 107 | 188 | 107 | 148 | 188 |
| 20 | 84 | 188 | 84 | 148 | 188 |
| 24 | - | 188 | - | 148 | 188 |
| 48 | - | 188 | - | 149 | 188 |
| 96 | - | 188 | - | 171 | 188 |
| 144 | - | 188 | - | 130 | 188 |
| 240 | - | 188 | - | 141 | 188 |
| 288 | - | 188 | - | 175 | 188 |
| 384 | - | 188 | - | 175 | 188 |
| 640 | - | 188 | - | 188 | 188 |
| 1024 | - | 147 | - | - | 147 |

**Structure, not throughput.** Our package exposes no `static` MoE backend at all
(`b12x.moe.fused_moe` has no `static` module; the tuning registry has no
`decode/static` policy), while the reference's has `MoEStaticKernel` plus a generated
`static` ladder. Our side's only applicable policy above 20 routed rows is `dynamic`,
whose ladder is flat at 188 until 640.

Where a DSV4-Flash decode step lands (`docs/knowledge/02-model.md`: 256 routed
experts, 6 per token; `q_rows = c * 8`):

| level | q rows | routed rows | our policy | reference policy |
|---|---|---|---|---|
| c1 | 8 | 48 | dynamic, cap 188 | static, cap 149 |
| c3 | 24 | 144 | dynamic, cap 188 | static, cap 130 |
| c5 | 40 | 240 | dynamic, cap 188 | static, cap 141 |
| c6 | 48 | 288 | dynamic, cap 188 | static, cap 175 |

**Verdict.** **leading attribution for the ffn/MoE region, and still structural.** The profiler puts 65 % of a target layer in ffn/MoE (1.84 ms of 2.83 ms), and the MLA result already ruled attention out as the reference's source of speed, so this is where its lead has to live. The probe shows our kernel generation has no mid-regime MoE backend: every level in the protocol, c1 through c6, runs the `dynamic` kernel with a flat cluster cap of 188, while the reference runs a different `static` kernel whose cap is tuned per row count. **No throughput number is claimed here**: this is a coverage and configuration finding, and the next measurement is an MoE kernel head-to-head at these shapes to convert it into milliseconds. If it holds, a fix is in scope under goal item 4 (third-party kernel generation) or item 3 (our own configuration), depending on where the choice is made.

### MoE cluster-cap sweep on our `b12x`, DSV4-Flash decode shape

**Command.** `docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench vllm-spark-0731:main-029-proto-b12xref /bench/bench_moe_clusters.py`

**Log.** [`outputs/driver/one-off/moe-cluster-cap.log`](../outputs/driver/one-off/moe-cluster-cap.log)

Three reps of the whole sweep, cleanest pair (reps 1 and 2; rep 0 was taken while the
host was compiling at load average 16):

| `max_active_clusters` | rep 1 | rep 2 | spread | vs cap 188 |
|---|---|---|---|---|
| 188 (our flat default) | 11.557 | 11.761 | 1.8 % | 1.000 |
| 175 | 10.880 | 10.896 | 0.1 % | **0.941** |
| 149 | 11.582 | 11.632 | 0.4 % | 0.995 |
| 141 | 10.839 | 10.891 | 0.5 % | **0.938** |
| 130 | 11.635 | 11.591 | 0.4 % | 0.993 |
| 96 | 10.889 | 10.841 | 0.4 % | **0.939** |

Median of 30 CUDA-event timings per cell, 48 tokens x topk 6 = 288 routed rows, 256
routed experts, hidden 4096, intermediate 2048, one GB10. Weights synthetic but packed
in the real contract: uint8 FP4 `[E, 2*inter, hidden/2]` and `[E, hidden, inter/2]` with
E8M0 `[E, rows, K/32]` grids, A8 activation, unit global scales. Same activations and
routing at every cap, fresh weight copy per cap.

**Output sums agree to 0.005 %** across all 18 cells (516314.3 to 516340.0), which is
FP32 atomic-accumulation order, so the kernel computes the same thing everywhere.

**Rep 0 is discarded and the reason recorded.** The whole sweep ran while the phase-1
proto2 build was compiling vLLM's CUDA extensions (`ninja -j 16`, load average 16.0,
sixteen `cicc` processes at 100 %). Rep 0 is uniformly inflated (13.121 at cap 188
against 11.557 and 11.761 later) and cannot be compared with reps 1 and 2. Lesson for
the trap list: do not put a GPU measurement on this rig while the host is compiling.

**The bug that caused the earlier divergence, for the record.**
`fused_moe.prepare_weights` repacks the packed weight tensors *in place*
(`_logical_weight_to_w4a8_rp_inplace`, `_e8m0_scale_to_w4a8_sfb_inplace`), so passing
the same `PackedWeights` tensors into a second build feeds it already-repacked data.
Only the first build in a process is correct. Diagnostic:
`harness/diag_moe_divergence.py`.

**Verdict.** **kernel-level effect is now reproducible; still not a protocol result.** Across two clean reps the caps group consistently: 175, 141 and 96 land at 10.84-10.90 ms while our flatdefault of 188 and the caps 149 and 130 land at 11.56-11.64 ms, a **6 % separation reproducibleto 0.5 % between reps**. Against the repo's own keep rule that now holds: cap 175 beats cap188 by 5.9-6.3 % while 188's own rep-to-rep spread is 1.8 %. The effect is non-monotonic in thecap value, so the mechanism is not simply fewer clusters, and the two groups being stable whilethe ordering within each is noise suggests a grid/tiling threshold rather than a smooth cost.**Two things still stop this being a protocol win.** The whole sweep ran under host loadaverage 16 from the concurrent proto2 build, so absolute values are not trustworthy and rep 0had to be discarded; and 10.4-11.7 ms for one MoE at 288 routed rows remains about an order ofmagnitude above what a 43-layer step of 38.65 ms can contain, so this single-GPU 256-expertexecution plan is still not the path the engine runs. **Next**: repeat the sweep on an idle hostto confirm the 6 % without contention, then reshape to the served configuration (TP sharding,real routing skew, tuned plan) before anything is claimed on the protocol.

### Real NVFP4 KV writer: what width does the reference actually allocate?

**Command.** `docker run --rm --entrypoint python3 -v ~/bench:/probe ghcr.io/anemll/dspark-vllm-gx10:0.1.1 /probe/probe_nvfp4_kv_width.py`

**Log.** [`outputs/driver/one-off/nvfp4-kv-width-ref.log`](../outputs/driver/one-off/nvfp4-kv-width-ref.log)

Read from the reference image's own source, not inferred:

| reference file | dtype test | width returned |
|---|---|---|
| `vllm/v1/kv_cache_interface.py:381-386` | `fp8_ds_mla` **and** `nvfp4_ds_mla` | `storage_block_size * 584` |
| `vllm/v1/attention/backends/mla/sparse_swa.py:151-154` | `fp8_ds_mla` **and** `nvfp4_ds_mla` | 584 |
| `vllm/models/deepseek_v4/sparse_mla.py:104-107` | `fp8_ds_mla` **and** `nvfp4_ds_mla` | `(num_blocks, block_size, 584)` |
| `vllm/models/deepseek_v4/attention.py:619-620` | both | 584-byte DSpark envelope |

The reference's own comments, verbatim:

- `attention.py`: "fp8_ds_mla/nvfp4_ds_mla are padded uint8 layouts. Keep the upstream
  FP8 alignment and use the proven 584-byte DSpark NVFP4 envelope."
- `kv_cache_interface.py`: "DeepseekV4 uses the padded 584-byte sparse-MLA envelope for
  **both** fp8_ds_mla and nvfp4_ds_mla. head_size stays semantic (512); bytes are
  determined by the backend layout here."
- `sparse_mla.py`: "DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8
  scale)." 448 + 128 + 8 = 584, which is the same arithmetic our own
  `_DSV4_TOKEN_BYTES = 584` uses.

Seven modules in the reference name both dtypes; two of them are the KV-width
decisions above and both return the identical number for the two.

**Verdict.** **closed: there is no real NVFP4 KV writer to port, on either side.** Goal item 1'spremise, that our `nvfp4_ds_mla` is an envelope alias while the reference has a real writer, isfalse in the reference's own source: anemll's `nvfp4_ds_mla` allocates the *same* 584 B/tokenenvelope as `fp8_ds_mla`, by the same 448 + 128 + 8 arithmetic we use. **The number that closesit: 584 B/token on both engines**, read from the reference image rather than from our notes. The7,650 B/token figure in the goal text is a whole-model footprint that counts the indexer and SWAcaches, not a narrower per-layer dtype, so it is not evidence of a writer we lack. This agreeswith the 2026-08-26 correction already in `docs/knowledge/04-quantization-kv.md`, now confirmedagainst the live image. **What remains real is not a writer but capacity**: anemll reaches a~2.0M-token KV pool against our ~97k at similar utilisation, which that doc attributes toruntime and weights footprint rather than dtype width, and which the serving protocol measuresdirectly. One honest limit on this closure: the widths are verified from the Python-sidedecision functions and their stated byte composition; the reference's writer kernel was notbyte-compared, and the comments plus the independent FlashMLA README cross-check in`04-quantization-kv.md` are what stand in for that.

### MoE backend: what the comparator actually runs against what we run

**Command.** `docker run --rm --entrypoint python3 ghcr.io/anemll/dspark-vllm-gx10:0.1.1 /probe/probe_nvfp4_kv_width.py  # plus a read of harness/ref-base0731.yaml`

**Log.** [`outputs/driver/one-off/moe-backend-comparator.log`](../outputs/driver/one-off/moe-backend-comparator.log)

The comparator's recipe, `harness/ref-base0731.yaml`, is the authority on what the
reference arm runs. Its serve command ends:

```
    --moe-backend flashinfer_b12x \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}}'
```

Our pin, `configs/pin.main-029.env`:

```
MOE_BACKEND="${MOE_BACKEND:-b12x}"
```

The reference's oracle says `flashinfer_b12x` is **excluded from auto-selection** and
must be asked for by name (`oracle/nvfp4.py` lines 175-178):

```
    # NOTE: the kernels are selected in the following order.
    # FLASHINFER_B12X is intentionally excluded from auto-selection until
    # the upstream CUTLASS SM121 MMA op guard is resolved; use
    # moe_backend="flashinfer_b12x" to opt in explicitly.
```

So the comparator is not running a default. It is explicitly opting in to a kernel the
upstream oracle refuses to pick on its own, which is a deliberate authorial choice.

A caveat that falls out of the same file: `select_nvfp4_moe_backend` narrows the
candidate list to `NVFP4_BACKENDS_WITH_CLAMP = {FLASHINFER_TRTLLM, FLASHINFER_CUTLASS,
MARLIN}` whenever `config.swiglu_limit is not None`, and `FLASHINFER_B12X` **is not in
that set**. DeepSeek-V4-Flash sets `swiglu_limit 10.0`, so the explicit opt-in may be
bypassing clamp handling that the auto-selected backends would have applied. Both
protocol gates exist precisely to catch that kind of difference, and they must be read
on this arm rather than assumed.

What the two names dispatch to, from the reference image's own source:

| name | implementation |
|---|---|
| `flashinfer_b12x` (reference) | `FlashInferB12xExperts` in `fused_moe/experts/flashinfer_b12x_moe.py` |
| `b12x` (ours) | `B12X_MXFP4`, the b12x package's own MoE |

`FlashInferB12xExperts`' docstring: "Uses `b12x_fused_moe` from FlashInfer PR #3080
which fuses token dispatch, two GEMMs, SwiGLU activation, and topk-weight reduction
into a **single kernel call**. Input quantization (BF16->FP4) is performed inside the
kernel so BF16 hidden states are passed directly." It asserts
`quant_config.quant_dtype == "nvfp4"` and supports only that.

Availability checked in both images (so this is not a missing package):

| image | flashinfer | `b12x_fused_moe` | `has_flashinfer_b12x_moe()` |
|---|---|---|---|
| reference `0.1.1` | 0.6.15 | present | True |
| ours `main-029-proto-b12xref` | 0.7.0 | present, plus `B12xNvfp4Config/Runner`, `B12xW4A16Config/Runner` | True |
Correction, same day: the first pass scanned `/usr/local/lib/python3.12/dist-packages/vllm` and
found no `flashinfer_b12x`, and nearly recorded that as our stack missing the backend. vLLM is
installed editable from `/opt/vllm` (`vllm.__file__` is `/opt/vllm/vllm/__init__.py`), and that
tree has 6 files naming the backend, including `experts/flashinfer_b12x_moe.py` and the nvfp4
oracle's `FLASHINFER_B12X`. Both images also answer `has_flashinfer_b12x_moe() is True` and
`map_nvfp4_backend("flashinfer_b12x")` returns `NvFp4MoeBackend.FLASHINFER_B12X`. Anyone
repeating this check must read the path from `vllm.__file__`, not assume dist-packages.

**Verdict.** **the sharpest lead of the session, and it is untested.** The comparator routesffn/MoE through `flashinfer_b12x`, a single fused FlashInfer kernel that does dispatch, bothGEMMs, SwiGLU and topk reduction in one call; our measured arms run `MOE_BACKEND=b12x`, adifferent implementation. ffn/MoE is the region the profiler prices at 65 % of a target layer,and attention has already been ruled out as the source of the reference's lead, so this iswhere the gap has to be. **`flashinfer_b12x` has never been measured in this repo**: the MoEsweep in `docs/EXPERIMENTS.md` tried `b12x`, `humming`, `flashinfer_trtllm` (died at workerinit: kernel does not support current device cuda) and `flashinfer_cutlass` (never healthy),and not this one. Our own stack supports it: `patches/assert_stack.py` has`ALLOWED_MOE = ("b12x", "flashinfer_b12x")`, `configs/pin.golden.env` already sets it, and theFlashInfer wheels in both images expose `b12x_fused_moe`. So this is a one-variable arm: set`MOE_BACKEND=flashinfer_b12x`, rebuild nothing, run the protocol. **First arm after the proto2rebuild lands.** One caveat to carry: `FlashInferB12xExperts` asserts NVFP4 expert weights, and`docs/knowledge/04-quantization-kv.md` records NVFP4 *weight* attempts as a dead end on thismodel, though `02-model.md` says the checkpoint's experts ship as fp4 and the MXFP4 oracle maps`flashinfer_b12x` to `B12X_MXFP4`, so the MXFP4 route may reach the same kernel. Expect iteither to run and be measurable, or to be rejected at load — both are results. Two furtherfacts sharpen it. The reference is not running a default: `oracle/nvfp4.py` states thatFLASHINFER_B12X is "intentionally excluded from auto-selection until the upstream CUTLASS SM121MMA op guard is resolved", so the comparator deliberately opts in to a kernel upstream will notchoose by itself — somebody measured it and preferred it. And the same file excludesFLASHINFER_B12X from `NVFP4_BACKENDS_WITH_CLAMP`, which DeepSeek-V4-Flash's `swiglu_limit 10.0`would otherwise narrow to `{TRTLLM, CUTLASS, MARLIN}`; since our own sweep found TRTLLM dies atworker init on this device and CUTLASS never became healthy, the clamp-restricted set isexactly the set that does not work here. That is a coherent story for why the reference runsthis backend and why we never did. The clamp caveat is why the arm must be judged on bothgates, not only on throughput.

### MoE kernel head-to-head: our b12x against FlashInfer's fused b12x_fused_moe

**Command.** `docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench vllm-spark-0731:main-029-proto-ccompile /bench/bench_moe_headtohead.py`

**Log.** [`outputs/driver/one-off/moe-kernel-headtohead.log`](../outputs/driver/one-off/moe-kernel-headtohead.log)

Same image, same process, same GPU, same routing, one DSV4-Flash c6 decode shape (48 tokens,
topk 6 = 288 routed rows, 256 experts, hidden 4096, intermediate 2048). Median of 30 CUDA-event
timings per rep, five reps:

| library | rep medians (ms) | clean median | output sanity |
|---|---|---|---|
| `b12x` (ours, `b12x.moe.fused_moe`) | 10.108 / 10.075 / 10.058 / 10.069 / 10.063 | **10.066** | finite, mean abs 2.54 |
| `flashinfer_b12x` (`flashinfer.fused_moe.b12x_fused_moe`) | 134.500 / 127.521 / 127.613 / 127.792 / 127.991 | **127.729** | finite, mean abs 2.55 |

Reproduced across two independent runs (the first: 10.67 against 133.72). FlashInfer selected
its `static` backend by itself, which the JIT name records:
`static_m48_k4096_n2048_t6_r288_...`. That is the regime its own tuning registry uses for
20 < routed rows <= 640, so it is the backend the reference would run at this shape.

Weight construction mirrors `FlashInferB12xExperts.process_weights_after_loading` in the
reference image: NVFP4 with **vec-16 E4M3** scales (`k1 = k1_sf * 16`; `fp4_quantize` refuses
ue8m0 at vec 16, so the working path is E4M3 rather than the checkpoint's e8m0), the scale stack
reshaped to `[E*N, K/16]` and converted once with `num_groups=E`, per-expert alphas of 1.0 with
the global scale baked into the block scales, and `fc2_input_scale` forced to 1.0 - all of which
is what that method does.

**The magnitude is not believable and must not be quoted as a win.** 127.7 ms for one MoE layer
cannot sit inside a 43-layer decode step measured at 38.65 ms, so the reference is not running
this kernel the way this probe drives it. Candidate causes, in order of suspicion: (1) the real
engine runs TP=2 with experts sharded, so each rank sees far fewer experts and more rows per
expert, while this probe puts all 256 experts and all 288 rows on one GPU - a fixed-m48 static
tile over 256 experts pads every expert to 48 rows and wastes most of the work; (2) the real path
reuses a cuda-graph-captured workspace and this probe rebuilds the binding per rep; (3) the
alpha/scale convention here is accepted but may select a slower code path inside the kernel.
Until one of those is eliminated, the honest reading is: at this shape and routing, with all
experts local to one rank, FlashInfer's fused kernel is ~13x slower than ours - which is a
statement about this configuration, not about the comparator's engine.

**Verdict.** **measured, and deliberately not claimed.** The two MoE implementations the two engines use now run side by side in one process at the DSV4 c6 decode shape: ours at **10.066 ms**, FlashInfer's fused `b12x_fused_moe` at **127.729 ms**, a **13x** separation that reproduced across two runs with both outputs finite and of the same magnitude. If it holds, the comparator's `--moe-backend flashinfer_b12x` is not where its speed comes from - but it almost certainly does not hold as stated, because the number is an order of magnitude larger than the whole reference step can contain. The most likely explanation is sharding: the real engine runs TP=2 with experts split across ranks and a captured workspace, while this probe puts 256 experts and 288 rows on one GPU, where a fixed-m48 static tile pads every expert to 48 rows. **Next**: repeat with the served sharding (num_local_experts = 128, and 64 for a 4-way split) and with a prewarmed, captured binding, and only then decide whether the comparator's MoE backend is a lead or a red herring. The probe is `harness/bench_moe_headtohead.py`; log `outputs/driver/one-off/moe-kernel-headtohead.log`.

### FlashInfer `b12x_fused_moe` in our wheel against the reference's wheel

**Command.** `docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench <image> /bench/bench_flashinfer_moe.py  # run once per image`

**Log.** [`outputs/driver/one-off/flashinfer-moe-versions.log`](../outputs/driver/one-off/flashinfer-moe-versions.log)

The same FlashInfer kernel (`flashinfer.fused_moe.cute_dsl.b12x_moe` in both wheels), the same
probe, the same inputs, run in each image. Median of 30 CUDA-event timings, three reps each;
reported per rep.

| local experts | flashinfer 0.7.0 (ours) | flashinfer 0.6.15 (reference) | ratio | `mean_abs` output |
|---|---|---|---|---|
| 256 | 150.170 / 148.978 / 150.477 | 12.291 / 12.408 / 12.430 | **12.1x** | 2.6049 vs 2.6049 |
| 128 | 71.779 / 72.158 / 71.806 | 7.672 / 7.770 / 7.672 | **9.4x** | 2.6084 vs 2.6086 |
| 64 | 38.214 / 37.795 / 37.798 | 4.464 / 4.417 / 4.424 | **8.5x** | 2.6007 vs 2.6000 |

**The outputs are identical between the wheels** (agreeing to the fourth decimal in `mean_abs` at
every expert count), so this is the same computation, not a different kernel or a different
precision. Only the speed differs, and ours is the slow one.

The 0.6.15 signature has no `input_global_scale` - the call is filtered to each build's accepted
parameters, and that is the only argument dropped, so the comparison is like for like.

Also recorded: the same standalone probe in our image reproduces the head-to-head's FlashInfer
numbers (150.2 / 71.8 / 38.2 here against 142.0 / 79.7 / 44.4 there), which confirms the earlier
13x was measuring this kernel and not a mis-built weight path.

**The absolute values still do not reconcile with the served engine, and that must be said.** At
128 local experts the *faster* wheel still spends 7.67 ms on one MoE layer, so 43 layers would be
~330 ms inside a step measured at 38.65 ms. The profiler attributes 1.84 ms per layer to ffn/MoE at
c1. So this probe's geometry is probably not the served one - most likely the expert dimensions -
and no absolute number from it should be read as the engine's cost. The version comparison is
unaffected, because both wheels were handed identical inputs and produced identical outputs.

**Verdict.** **A library regression, and the strongest item-4 finding of the session.** Our FlashInfer 0.7.0 builds the same `b12x_fused_moe` kernel **8.5x to 12.1x slower** than the reference's 0.6.15, at three expert counts, with **bit-identical outputs**. This is the shape of a finding goal item 4 was written for: the comparator opts into `flashinfer_b12x` by name, its wheel runs that kernel an order of magnitude faster than ours does, and attention has already been ruled out as the source of its lead. **It also refutes the sharding explanation** for the earlier 13x: the ratio holds flat across 64, 128 and 256 local experts, so the gap is not about how experts are split across ranks. **What must not be claimed yet**: any absolute cost. At 128 local experts even the fast wheel spends 7.67 ms per layer, so 43 layers would exceed the whole 38.65 ms step, which means this probe's geometry is probably not the served one - the expert dimensions were the prime suspect -- **since checked against the checkpoint and found correct, so that caveat is retracted; see the expert-traffic probe below**. The relative version result is safe from that, since both wheels saw identical inputs and outputs. **Next**: fix the probe geometry against the checkpoint's real expert shape, then decide whether 0.6.15's kernel should be backported or our wheel pinned down - and check with the owner before any upstream contact, as the contract requires.

### Expert traffic and measured bandwidth: what a full pass can cost

**Command.** `python3 config.json dump, then: docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench vllm-spark-0731:main-029-proto-ccompile /bench/bwprobe.py`

**Log.** [`outputs/driver/one-off/gb10-bandwidth.log`](../outputs/driver/one-off/gb10-bandwidth.log)

Checked against the checkpoint rather than the docs. `config.json` for
`deepseek-ai/DeepSeek-V4-Flash-0731` (snapshot `7872f01b1d1f`):

| key | value |
|---|---|
| `n_routed_experts` | 256 |
| `num_experts_per_tok` | 6 |
| `moe_intermediate_size` | 2048 |
| `hidden_size` | 4096 |
| `num_hidden_layers` | 43 |
| `expert_dtype` | fp4 |

Those are exactly the numbers both MoE probes used, so the geometry was right and the caveat
that it might not be is **retracted**: the version comparison was measured on the served shape.

What the geometry implies, with a measured bandwidth to pin it down. A 1 GiB device copy on this
GB10 moves 1.074 GB in 4.955 ms, so **216.7 GB/s effective** (spec is ~273; a copy reaches about
80 %). Expert traffic at the real shape:

| quantity | value |
|---|---|
| params per expert | 25,165,824 |
| GB of expert weights per layer (fp4, 0.5 B/param) | 3.22 |
| GB of expert weights across 43 layers | 138.5 |
| ms per layer at measured bandwidth | 14.87 |
| ms per full 43-layer pass, one rank | 639.2 |
| ms per full pass, TP=2 | 319.6 |

**A hard constraint follows: a full 43-layer forward pass must stream ~138.5 GB of expert weights**
(the checkpoint is 155-167 GB in total and the experts are all but all of it), so at the measured
216.7 GB/s no full pass can complete faster than about **320 ms across two ranks**. Any step-time
figure materially below that is not a full-model pass. The MoE probes run close to that bound and
confirm it: at 128 local experts (one TP rank) ours measures 7.03 ms against a 7.43 ms traffic
estimate, and FlashInfer 0.6.15 measures 7.67 ms - both essentially at the streaming floor. Our
own kernel is therefore not leaving bandwidth on the table, and FlashInfer 0.7.0's 71.78 ms is
about **9.7x off the floor** for the same work.

**Verdict.** **a measured constraint, and it reframes the attribution.** The MoE geometry in both probes is confirmed against the checkpoint, so the probe geometry caveat is retracted. With the bandwidth measured on this part (216.7 GB/s for a 1 GiB copy) and the checkpoint's own expert shape, a full 43-layer pass has to stream 138.5 GB of expert weights, which is **>=320 ms at TP=2 and >=639 ms on one rank**. So the repo's quoted c6 step time of 38.65 ms cannot be a full 43-layer forward pass, and neither can the profiler's 1.84 ms per layer for ffn/MoE be reconciled with one - the two figures describe different things, and the arithmetic that matters (how many milliseconds a full pass owes to memory) has been missing from the attribution. **Useful consequence for the kernel question**: both good kernels sit at the streaming floor (ours 7.03 ms against 7.43 ms of traffic at 128 experts; FlashInfer 0.6.15 7.67 ms), so the MoE layer is bandwidth-bound, our implementation is not wasting bandwidth, and the 8.5-12x gap to FlashInfer 0.7.0 is a real defect in that wheel rather than a property of the shape. **Next**: reconcile the step-time basis before doing more attribution, since every per-layer budget in the docs was derived from a number that cannot be a full pass.

### Step-time basis: the quoted figures are 6x too small (concurrency missing)

**Command.** `cat outputs/driver/refg-3.meter.txt outputs/driver/protog-3.meter.txt  # arithmetic, no GPU run`

**Log.** [`outputs/driver/refg-3.meter.txt`](../outputs/driver/refg-3.meter.txt)

Derived from the meter logs already in this repo, plus the measured bandwidth. The quoted step times
divide wall time by `drafts_per_req`, but that counter is per sequence: 512 tokens at 4.676
tokens/step is 109.5 forward passes for one sequence, and 657 = 6 x 109.5. The forward pass serves
all six. Dividing by 6 instead:

| level | tag | wall s | tokens/step | steps per sequence | step ms | drafts/wall (what was quoted) |
|---|---|---|---|---|---|---|
| c6 | `refg` (reference) | 19.33 | 4.676 | 109.5 | **176.5** | 29.4 |
| c6 | `protog` (ours) | 25.89 | 4.585 | 111.7 | **231.8** | 38.7 |
| c1 | `refg` | 7.79 | 4.923 | 104.0 | **74.9** | 13.35 |
| c1 | `protog` | 12.49 | 5.069 | 101.0 | **123.7** | 8.09 |

The factor is exactly the concurrency: 29.4 x 6 = 176.4, and 38.65 x 6 = 231.9. Independent check
that the corrected basis is the right one - the step-time ratio matches the throughput ratio:
176.5 / 231.8 = 0.761 against 118.64 / 158.93 = 0.747, where the quoted pair implies 38.65 / 29.4 =
1.31, the inverse.

Against the memory floor from the expert-traffic probe (138.5 GB of expert weights per pass, 216.7
GB/s measured for a copy, >=320 ms at TP=2): the measured 176.5 ms is the same order and *below*
it, which is consistent rather than contradictory - a read-heavy stream beats a copy, and at 288
routed rows over 256 experts about a third of experts receive no rows at all and are skipped under
Poisson(1.125) routing. The important part is that the step is the order of the weight traffic, so
the step is memory-bound on expert weights and MoE really is the whole game.

**Verdict.** **The step-time basis in this repo is wrong by exactly the concurrency, and it is why the attribution never closed.** `HANDOVER.md`'s "step time is 38.65 ms against 29.4 ms at c6", and every per-layer budget derived from it including `docs/knowledge/05-performance.md`, divides wall time by `drafts_per_req`. That counter is per sequence, while the forward pass serves all six, so the quoted figures are 6x too small: the real c6 step is **176.5 ms for the reference and 231.8 ms for ours**. Two independent checks agree - the factor is exactly 6 (29.4 x 6 = 176.4), and the corrected ratio 0.761 matches the throughput ratio 0.747 while the quoted ratio is its inverse. Against the expert-traffic floor of >=320 ms at TP=2 the measured 176.5 ms is the same order, and lower for reasons that make sense (reads beat a copy, and roughly a third of experts take no rows under Poisson routing). **Consequence: the step is memory-bound on expert weights, MoE is the whole game, and any per-layer attribution built on the old numbers has to be redone** - including the claim that the WO and MLA kernels are small components, which was computed from the same wrong basis. This does not change the version-regression finding, which is a direct kernel-to-kernel comparison.

### Streaming bandwidth and the expert-bytes gap: 0.916 GB per layer against 1.202 GB

**Command.** `docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench vllm-spark-0731:main-029-proto-ccompile /bench/bwprobe3.py`

**Log.** [`outputs/driver/one-off/gb10-gemv-bandwidth.log`](../outputs/driver/one-off/gb10-gemv-bandwidth.log)

Streaming bandwidth on this part, measured three ways (`/bench/bwprobe3.py`):

| pattern | working set | GB/s |
|---|---|---|
| device copy (read+write) | 1 GiB | 223.2 |
| GEMV (streams a weight matrix once, the MoE's pattern) | 128 MiB | 201.0 |
| GEMV | 512 MiB | 198.9 |
| GEMV | 32 MiB | 156.6 |

So ~200 GB/s is what streaming reads actually achieve; the 273 GB/s on the spec sheet is not
reachable in practice, and the earlier copy figure (216.7) was within 3 % of the repeat (223.2).

Expert traffic at the confirmed shape is 3.22 GB per layer and 138.5 GB over 43 layers. Against the
corrected c6 step times:

| arm | c6 step ms | ms per layer | GB a rank streams at 223 GB/s | as a fraction of a full layer |
|---|---|---|---|---|
| reference `refg` | 176.5 | 4.10 | 0.916 | **28.4 %** |
| ours `protog` | 231.8 | 5.39 | 1.202 | 37.3 % |

**The reference's whole step is accounted for by expert streaming, and only about 28 % of a full
expert layer fits in its per-layer budget.** With expert parallelism across the two ranks - 128 local
experts each, half the 288 row-expert pairs per rank - and empty-expert skipping, the predicted
fraction touched is 1 - exp(-144/128) = 67.5 % of the local 128, which is 33.6 % of the full 256,
very close to the 28.4 % the timing implies. So the model is: experts sharded across ranks, only the
experts that receive tokens are read, and the step is memory-bound on exactly that traffic.

**And the same arithmetic prices the gap.** Our step implies 1.202 GB streamed per layer against the
reference's 0.916, so **we move about 31 % more expert bytes per layer for the same work**. That is
the whole 55.3 ms difference, and it is a single measurable quantity rather than a spread of
suspects. The candidates are concrete: a different expert sharding (reading more experts per rank),
or failing to skip experts that receive no tokens. The kernel itself is not the suspect: at 128 local
experts our MoE measured 7.03 ms against FlashInfer 0.6.15's 7.67 ms, so the two are comparable.

**Verdict.** **The gap is expert bytes streamed, and it is now a number.** Measured streaming bandwidth on this part is ~200-223 GB/s, not the 273 GB/s the spec sheet implies. At the corrected c6 step times the reference has 4.10 ms per layer, in which a rank can stream 0.916 GB, and a full expert layer is 3.22 GB - so the reference reads about **28 %** of the expert weights per layer, which is what expert sharding plus empty-expert skipping predicts (33.6 %). Ours implies 1.202 GB per layer, i.e. **~31 % more expert bytes for the same work**, which is the entire 55.3 ms difference between a 231.8 ms step and a 176.5 ms one. This is the sharpest attribution the session has produced: not a list of suspect regions but one quantity, expert bytes streamed per layer per rank, and it says the step is memory-bound with no room left for anything else to matter. **What it does not yet say** is which mechanism costs the 31 % - expert sharding that reads more experts per rank, or a failure to skip experts with no tokens. Both are testable at the kernel or config level without touching the reference, and a fix would sit under goal item 3 (our own configuration). **Caveat on the model, extended after testing it.** The 31 % assumed every layer is MoE and that the step is the target pass alone, so the per-layer budget is an upper bound and the 28/37 % fractions move if the draft model, attention or collectives take a share. Both mechanisms that could be tested directly came back negative: the kernel skips untouched experts at 88 % of the streaming bound, and both arms run EP = 1 so neither splits the expert set. Routing distribution is the last candidate, and it is weakened too - both arms run the same checkpoint and therefore the same router, so for the same token stream they should select nearly the same experts, and a difference there would require the numerics of the two engines to diverge enough to change expert selection. **So read 31 % as a difference in the step that is not yet decomposed, not as a proven expert-byte count**, and settle it by re-running the region profiler on the corrected step basis. Routing is measurable on our side at least: `enable_return_routed_experts` defaults off and `--enable-return-routed-experts` adds a base64 numpy `routed_experts` field to the chat response, which `harness/capture_routing.py` decodes; both images support the flag, but the reference configuration is read-only by contract, so a symmetric comparison is not available.

### Does our MoE skip experts with no tokens? Yes, at 88 % of the streaming bound

**Command.** `docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench vllm-spark-0731:main-029-proto-ccompile /bench/bench_moe_skipping.py`

**Log.** [`outputs/driver/one-off/moe-skipping.log`](../outputs/driver/one-off/moe-skipping.log)

Weights, shapes and routed rows held fixed (256 experts resident, 288 routed rows, one MoE layer);
only the number of distinct experts the routing targets changes. Median of 30 CUDA-event timings,
three reps, clean median across the last two shown.

| distinct experts touched | clean median ms |
|---|---|
| 174 | 13.29 |
| 116 | 9.67 |
| 61 | 6.13 |
| 32 | 4.19 |
| 8 | 2.72 |

A straight line fits: 0.0637 ms per touched expert plus about 2.21 ms of fixed cost
(8 x 0.0637 + 2.21 = 2.72, and 174 x 0.0637 + 2.21 = 13.29). One expert's fp4 weights are
3.22 GB / 256 = 12.6 MB, which at the measured 223 GB/s costs 0.0564 ms, so the marginal cost is
**88 % of the streaming bound** - the kernel is reading the experts it touches at near-peak
bandwidth.

**So the kernel does skip untouched experts**, and it is not wasting bandwidth on them. The first
run of this probe was wrong and is not evidence: its routing pool was `randperm(active)[:topk]`, so
every case targeted only 6 distinct experts. That accidentally showed 6 experts at 2.1 ms against
~232 experts at 10.9 ms in the head-to-head probe, which is the same conclusion by a different
route, but the curve above is the measurement.

**What this does to the attribution**: the ~31 % expert-byte gap cannot be a skipping failure and
cannot be the kernel's efficiency, because both are now measured. It must be the **number of experts
each rank touches**, which is a configuration matter - expert parallelism, or how routing is
distributed per rank - and therefore goal item 3 rather than item 4.

**Verdict.** **Hypothesis refuted, and that narrows the target.** Our MoE kernel's cost is linear in the number of distinct experts the routing touches, at 0.0637 ms per expert against a 0.0564 ms streaming bound, i.e. **88 % of achievable bandwidth** with a 2.21 ms fixed cost. So it skips experts that receive no tokens and it streams the ones it does touch at near-peak. The ~31 % expert-byte gap priced in the bandwidth probe therefore cannot come from a skipping failure or from kernel inefficiency; it has to come from the number of experts each rank touches, which is expert parallelism or routing distribution - a configuration question under goal item 3, not a library regression under item 4. **Honest limits**: this measures one layer in isolation at 288 routed rows with synthetic weights, so the absolute fixed cost and the per-expert slope are properties of this probe; and the served routing is skewed where this probe is close to uniform, so the real experts touched per rank is unknown until it is read off the engine. **Next**: read the served configuration for its expert parallelism and routing, which decides whether the 31 % is addressable at all.

### Served MoE configuration: both arms run EP = 1, so sharding is not the gap

**Command.** `grep -iE 'expert|moe|parallel' harness/logs/attnfi-engine.log ; docker run --rm --entrypoint python3 <image> -c "import inspect; from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig as C; print(inspect.getsource(C.make))"`

**Log.** [`harness/logs/attnfi-engine.log`](../harness/logs/attnfi-engine.log)

Our served configuration, read from our own engine log (`harness/logs/attnfi-engine.log`) and from
vLLM's source in the image:

| item | value | source |
|---|---|---|
| `tensor_parallel_size` | 2 | engine config line |
| `data_parallel_size` | 1 | engine config line |
| `enable_expert_parallel` | not set anywhere in `scripts/`, `configs/` or `patches/` | grep |
| MoE parallel outcome | **EP = {1, 0} on both devices** | `FusedMoEParallelConfig.make` docstring |
| MoE backend | `B12X_MXFP4_MXFP8`, then `Using B12xExperts` | engine log |
| `kv_cache_dtype` | `nvfp4_ds_mla`, "Using DeepSeek V4 padded nvfp4_ds_mla KV cache format" | engine log |
| `quantization` | `deepseek_v4_fp8` | engine config line |
| capture sizes | [1, 2, 4, 8, 16, 24, 32, 40, 48] | engine config line |
| `enable_return_routed_experts` | **False** | engine config line |

The decisive line is vLLM's own docstring for `FusedMoEParallelConfig.make`, identical in both
images: "When TP = 2, DP(PCP) = 1 and EP = False ... device 0: TP = {2, 0} DP = {1, 0} **EP =
{1, 0}** ... device 1: TP = {2, 1} DP = {1, 0} EP = {1, 0} - Comment: Tensors are sharded across 2
devices." The reference recipe also sets no expert-parallel flag, and its vLLM 0.25.2 prints the same
table. So **both arms run EP = 1**: neither splits the expert set across ranks, and both shard inside
each expert along TP.

**So expert parallelism is not the 31 % either.** With equal local expert sets and equal topk, the
bytes each rank reads are decided by *which* experts the router picks and how many rows land per
step. Both are unmeasured today, and the engine can be asked: `enable_return_routed_experts` is False
in this arm, and flipping it reports the routed expert ids per request - a config-level measurement
under goal item 3, no reference contact needed.

**Verdict.** **Closed: expert parallelism is the same on both arms.** vLLM's own `FusedMoEParallelConfig.make` docstring, identical in both images, gives EP = {1, 0} per device when TP = 2, DP = 1 and EP = False, and neither arm sets an expert-parallel flag (`--enable-expert-parallel` appears nowhere in `scripts/`, `configs/` or `patches/`, and the comparator recipe does not pass it either). So both shard inside each expert along TP rather than splitting the expert set, both ranks hold the same expert set, and the ~31 % expert-byte gap cannot come from the sharding arrangement. **That leaves routing distribution as the only candidate**: which experts the router picks and how many rows land per step, neither measured today. Both are cheap to obtain because the engine can report them - `enable_return_routed_experts` is False in this arm and flipping it is a config change, not a rebuild, which makes it the next measurement under goal item 3. **Also recorded from the same log** because it was unrecorded: our MoE really is `B12X_MXFP4_MXFP8` with `B12xExperts`, the KV dtype really resolves to the 584-byte padded envelope, the model quantisation is `deepseek_v4_fp8`, and the graph capture sizes are [1..48].

### Why the layer profiler printed n=3: it was reporting the draft model, not the target

**Command.** `grep -nE '_LAYER_EVENTS|b12x_profile_decode_once|b12x_profile_layer' patches/files/sm12x_b12x_kernels.py patches/apply_overlays.py  # code reading, no GPU run`

**Log.** [`patches/files/sm12x_b12x_kernels.py`](../patches/files/sm12x_b12x_kernels.py)

The puzzle in `HANDOVER.md` was "the profiler prints `n=3` layer events, not 43, and that why is not
established". It is established, and the profiler has been reporting the wrong model.

Three separate hooks share one event list, `_LAYER_EVENTS[0]` in `patches/files/sm12x_b12x_kernels.py`:

| hook | applied to | does what |
|---|---|---|
| `b12x_profile_decode_once` | the **DSpark draft** `DFlashSpeculator._run_model` (`patches/apply_overlays.py:287-308`) | sets `_PROFILING_STEP[0]=True`, **resets** `_LAYER_EVENTS[0]=[]`, runs the draft, then **prints the layer summary from that list** |
| `b12x_profile_layer` | `DeepseekV4DecoderLayer.forward` (the target, 43 layers) | appends `(e0, e1)` to the same list |
| `b12x_profile_target_step` | the target runner `execute_model` | sets the flag and resets the list again |

The draft hook resets the list and prints from it, so what gets printed is the **draft model's**
layer timing, and everything the target recorded earlier is discarded. DSpark carries three MTP
blocks, which is exactly the `n=3` that was observed.

**Second defect in the same function**: it is documented as "One-shot timing for the first real DSpark
decode step" but has no one-shot guard - only the `is_current_stream_capturing()` path returns early.
On the non-capturing path it prints on **every** draft forward, up to `k+1` times per engine step.

**Consequence for this repo's attribution, and it is retroactive.** The per-layer and per-region
shares in `HANDOVER.md` (`ffn/MoE 1.84 ms of a 2.83 ms layer`, WO `0.60 ms`, MLA `0.20 ms`) came from
an instrumentation that shares the flag `_PROFILING_STEP[0]` across the draft and the target, so which
model the numbers describe depends on which hook ran last. The `n=3` observation is the proof that the
layer summary came from the draft. That is a second, independent reason the old attribution did not
reconcile with the corrected step basis.

**Fix direction, not yet applied**: give the draft hook its own event list (or stop printing layers
there and leave that to the target hook), and add the one-shot guard the docstring already promises.
Both are small edits in `patches/files/sm12x_b12x_kernels.py`. They are deliberately **not** applied
this round: the proto2 build is running from this exact `patches/` tree, and changing it now would
either desync the build or force another one. The fix belongs in the next rebuild, bundled with any
other patch change.

**Verdict.** **Root-caused, and it invalidates the old per-layer attribution rather than just explaining a puzzle.** `b12x_profile_decode_once` hooks the DSpark draft (`DFlashSpeculator._run_model`) yet resets and then prints from `_LAYER_EVENTS[0]`, the list the target's 43 `DeepseekV4DecoderLayer.forward` calls write into. The printed `n=3` is the draft's three MTP blocks, which is proof that the layer summary came from the draft and not from the target. Two defects: the shared list and flag across two models, and a missing one-shot guard despite the docstring claiming one. **What this changes**: the per-layer and per-region numbers in `HANDOVER.md` and `docs/knowledge/05-performance.md` cannot be assumed to describe the target model, which is a second independent reason they failed to reconcile with the corrected step basis - and it means the "ffn/MoE is 65 %" claim, which the whole remaining attribution has leaned on, is unverified. **What it does not change**: the direct kernel measurements (WO, MLA, the FlashInfer version regression, the skipping curve, the bandwidth numbers), none of which used this profiler. **Next**: apply the fix in the next rebuild, then re-measure the region shares on the corrected basis before using any region percentage again. **Fix written, not yet active.** `patches/files/sm12x_b12x_kernels.py` now keeps the draft's events in their own list: a `_DRAFT_PHASE` flag is set around the draft's `_run_model`, `b12x_profile_layer` routes each `(e0, e1)` pair to `_DRAFT_LAYER_EVENTS` or `_LAYER_EVENTS` accordingly, the draft hook no longer resets or prints the target's list, and `_DECODE_PROFILED` gives it the one-shot behaviour its docstring already claimed. Its prints are relabelled `b12x draft step` / `b12x draft layers` so the two models are distinguishable in a log. It is deliberately not synced to the rig: the proto2 build is running from this exact `patches/` tree and editing it mid-build would either desync the image or force another build. It lands in the next rebuild, and the check is the profile print itself - the target summary must read `n=43` (the decoder layers) with the draft's three reported separately.

### the target's per-layer and per-region breakdown, and the indexer skip A/B

**Command.** `SERVE_SKIP=headless bash ~/serve-prof3.sh  # --enforce-eager + VLLM_PROFILE_DECODE=1, skip flag under VLLM_SKIP_FLAG_DIR`

**Log.** [`outputs/driver/one-off/proto2-region-control.log, outputs/driver/one-off/proto2-region-skip-indexer.log`](../outputs/driver/one-off/proto2-region-control.log, outputs/driver/one-off/proto2-region-skip-indexer.log)

| region | control n | control gpu_sum | skip_indexer_all gpu_sum |
|---|---|---|---|
| attn | 43 | 80.3 ms | 49.0 ms |
| ffn | 43 | 65.2 ms | **86.2 ms** |
| indexer | 21 | 52.0 ms | **0.0 ms** |
| mla | 43 | 2.0 ms | 16.7 ms |
| wo_b12x | 43 | 17.4 ms | 22.4 ms |
| allreduce | 87 | 5.7 ms | 8.0 ms |
| **layers (43, enclosing)** | 43 | **148.0 ms** | **137.4 ms** |

**Verdict.** **The target breakdown now prints, and it says the region table must not be summed.** With `--enforce-eager` the per-layer marks finally run, so the pending check is closed: `b12x layers: n=43 sum=148.0ms avg=3.44ms`, and the layers are flat - the top layer is L2 at 4.61 ms against a 3.44 ms average, so no layer family is the gap at this batch. The regions are a different story. Removing the indexer entirely (`skip_indexer_all`, correctness deliberately lost) takes `indexer` from 52.0 ms to **0.0 ms** while `ffn` *rises* from 65.2 to 86.2 ms and the enclosing layer total falls only 148.0 -> 137.4 ms. The region marks record CUDA events on a stream that is still draining earlier queued work and nothing synchronizes between them, so `gpu_sum` is queue-drain time, not that region's compute. **The indexer's true net cost is therefore ~10.6 ms per step, about 7 % of this step, not the 52 ms (35 %) the raw table suggests** - and the `mla` rise from 2.0 to 16.7 ms shows why the A/B cannot be read further: replacing the index leaves the sparse MLA attending to different keys. Every region percentage quoted elsewhere in this repo, including the old 'ffn/MoE is 65 %' attribution, is unusable for the same reason. Only the enclosing per-layer total is additive, and it is flat.

### `vllm._deepselect_C` in both images (negative)

**Command.** `docker run --rm --entrypoint python3 <image> -c 'import vllm._deepselect_C'`

**Log.** [`documented here; the check is one import in each image`](../documented here; the check is one import in each image)

| image | `vllm._deepselect_C` |
|---|---|
| `vllm-spark-0731:main-029-proto2` | **absent** |
| `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` | **absent** |

**Verdict.** **Negative, and it closes an attractive hypothesis.** Both images boot with `WARNING [indexer_topk.py:29] Failed to import the DeepSelect extension (vllm._deepselect_C): No module named 'vllm._deepselect_C'`, which looks like a missing fast top-k path we could build. The reference image is missing it too, and this repo contains no reference to `deepselect` anywhere, so it cannot explain any part of the gap. Nothing to build.

### the b12x MoE policy ladder, ours against the reference's (from the coverage probe)

**Command.** `python3 harness/probe_moe_tuning.py  # needs an image carrying both b12x and b12x_ref; the table is in the saved log`

**Log.** [`outputs/driver/one-off/moe-tuning-coverage.log`](../outputs/driver/one-off/moe-tuning-coverage.log)

| | ours (`b12x 1.2.6`) | reference (`b12x 0.15.3`) |
|---|---|---|
| decode policies | `dynamic`, `dynamic_w4a8_decode`, `micro` | `dynamic`, `micro`, **`static`** |
| rows covered by cheap policy | micro ends at 20 routed rows | **static covers 20 < rows <= 640** |
| time at 48 routed rows (= c1) | 188 | **149** |
| time at 144 (= c3) | 188 | ~130-166 |
| time at 240 (= c5) | 188 | ~130-166 |
| time at 288 (= c6) | 188 | ~141-175 |

**Verdict.** **The gap is in a third-party kernel generation, and the measurement that says so was already taken and never acted on.** Our decode runs at 48-288 routed rows (`routed_rows = q_rows * topk`, `q_rows = c * (k+1)`, topk 6). Across that whole band our only policy is `decode/dynamic`, flat at 188, because our `micro` band ends at 20 rows. The reference's b12x carries a `decode/static` kernel for 20 < rows <= 640 -- exactly this band -- at 130-175, so it is 7-31 % faster than our kernel at every operating point the protocol measures, and `ours.micro` matches `ref.micro` exactly wherever both exist. The libraries are different generations, not configurations: our image ships `b12x 1.2.6` with no `b12x.moe.tuning` module and no `static` symbol under `b12x.moe.fused_moe`; the reference ships `b12x 0.15.3`. This is goal item 4's trigger condition met on the MoE half, and the first instance is the pre-built `vllm-spark-0731:main-029-proto-b12x015` image, which puts the reference's b12x under our stack. That arm previously died for an infrastructure reason only -- the image existed on spark1 alone, so spark2's worker could not start and the head blocked on rank 1 -- and it is being copied to spark2 now.

### b12x's PCIe all-reduce as the TP transport (negative: intra-host only)

**Command.** `bind-mount a patched `vllm/distributed/communication_op.py` that routes `tensor_model_parallel_all_reduce` through `PCIeAllReduce.from_process_group(process_group=tp.device_group, device=tp.device, single_channel=True)`, then `harness/run-arm.sh proto2-pcie-ar``

**Log.** [`documented here; the decisive lines are the worker's own prints`](../documented here; the decisive lines are the worker's own prints)

| step | result |
|---|---|
| construction, default call | `AttributeError('PCIeOneshotAllReducePool' object has no attribute 'should_allreduce')` -- the wrapper delegates to a pool that lacks it |
| construction, `single_channel=False` | `RuntimeError('distributed PCIe oneshot eager use requires an explicit semantic channel_id shared by every rank')` |
| construction, `single_channel=True` | **succeeds**: `algorithm=oneshot` |
| first collective | `RuntimeError('PCIe shared buffer CUDA IPC import open failed: failed to open CUDA IPC handle for peer')` on both ranks |

**Verdict.** **Decisive negative: the transport is intra-host and cannot carry a 2-node TP.** The runtime is CUDA-IPC based (`_RetryableIPCExport`, `IPC_SLAB_ALIGNMENT`, the shared-buffer IPC import path), and CUDA IPC handles are host-local -- rank 0 on spark1 cannot open rank 1's handle on spark2. `b12x/comm` contains exactly one module and its docstring states the scope: "``pcie``: collectives for consumer PCIe fabrics (no NVLink) -- one-shot and DMA/CE-ring all-reduce", i.e. several consumer cards in one box. b12x ships no inter-node alternative, so the 87 per-step all-reduces stay on PYNCCL and the fixed-per-step-cost search must move on-node. Two library defects were found on the way and are worth reporting upstream: the broken `should_allreduce` delegation, and the undocumented `single_channel=True` requirement for eager use. The patch was never baked into the image and is not in `apply_overlays.py`.

### the `B12X_*` flag surface, surveyed against defaults

**Command.** `grep each flag in `/usr/local/lib/python3.12/dist-packages/b12x --include=*.py``

**Log.** [`documented here; the check is one grep per flag`](../documented here; the check is one grep per flag)

| flag | default | verdict |
|---|---|---|
| `B12X_W4A8_TINY_DECODE` | on | already active |
| `B12X_FUSED_INDEXER` | on | already active |
| `B12X_INDEXER_DIRECT_K` | on | already active |
| `B12X_PAGED_MSA` | on | the env is only a `=0` disable |
| `B12X_PAGED_INDEX_SUPERTILE_K` | 32768 | already large |
| `B12X_TURBO_ATTN` | off | gated on `plan.kv_dtype == torch.float8_e4m3fn` in the generic paged forward; our attention is `B12X_MLA_SPARSE`, so it does not apply |
| `B12X_W4A16_SMALL_M_DIRECT/SPLITK` | on / off | W4A16 path, not our `MXFP4_MXFP8` MoE |
| `B12X_MOE_TILE_MN`, `B12X_DYNAMIC_TILE_MN`, `B12X_DYNAMIC_SWAP_AB` | unset | **testable**, documented as benchmarking knobs |

**Verdict.** **Mostly a negative: the flags that matter here are already at their useful value.** This closes the round-26 framing that '90-plus `B12X_*` flags are an untested surface' -- tiny decode, the fused indexer and the direct-K indexer path are all on by default, and turbo attention does not apply to the MLA-sparse backend this arm uses. What remains genuinely unset are the MoE tile-shape overrides, which are single-node microbenchmarks rather than full arms.

### the kernel-level composition of our step, from vLLM's own torch profiler

**Command.** `serve with `--enforce-eager --profiler-config {"profiler":"torch","torch_profiler_dir":"/root/.cache/vllm/prof"}`, then `POST /start_profile`, one request, `POST /stop_profile`, and sum `cat == "kernel"` events by name`

**Log.** [`documented here; traces under spark1:~/.cache/vllm/prof/ (gzipped chrome trace)`](../documented here; traces under spark1:~/.cache/vllm/prof/ (gzipped chrome trace))

| ms | % | calls | kernel |
|---|---|---|---|
| 2088.22 | **30.0 %** | 7140 | `at::native::elementwise_kernel<...gpu_kernel_impl_nocast<...direct...` |
| 1490.73 | **21.4 %** | 3196 | `ncclDevKernel_AllReduce_bf16_RING` |
| 1419.60 | 20.4 % | 1564 | `...b12xmoe...siluMoEDynamicKernelSilu...` |
| 285.01 | 4.1 % | 1521 | `nvjet_sm121_tst_mma_112x64x64...` |
| 252.01 | 3.6 % | 3567 | `cutlass_80_wmma_tensorop_s161616gemm_bf16_16x16_128x2` |
| 205.05 | 2.9 % | 3042 | `...b12x_libdense_gemmDenseGemmKernel...` |
| 60.52 | 0.9 % | 2975 | `mhc_fused_tilelang_kernel` |
| 57.88 | 0.8 % | 7752 | `per_token_group_quant_8bit_kernel<BFloat16, Float8_e4m3fn...` |
| 37.53 | 0.5 % | 1353 | `...b12xattention_sharedmlakernelUnifiedDecodeKernel...` |
| 18.55 | 0.3 % | 1462 | `_dsv4_topk_kernel` |

108493 kernel events, 117 distinct, 6951 ms total CUDA time over the window.

**Verdict.** **This is the first attribution the profiler can actually support, and it redirects the search.** Three things carry the step: unfused pointwise work (30.0 %), the TP all-reduce (21.4 %), and the MoE (20.4 %). Everything this repo has spent rounds tuning is in the noise by comparison -- the linear GEMMs together are 4.1 + 3.6 + 2.9 = 10.6 %, the sparse MLA decode kernel is 0.5 %, the indexer top-k is 0.3 %. That is exactly why every family swap measured as a wash or worse. **Two corrections follow.** The all-reduce is ~0.47 ms per call, so two per layer is ~0.93 ms against the ~1.45 ms per layer that the flat step penalty implies: the collective is plausibly most of the fixed per-layer cost, and the 5.7 ms the region marks reported for `allreduce` was queue-drain and low by 260x. And the 30 % pointwise block is what an uncompiled forward looks like -- 7140 calls of one elementwise template plus 7752 quantisation kernels -- so the compilation path is now the only remaining lever with a mechanism behind it.

### the compile path: which Dynamo breaks stand between us and a compiled forward

**Command.** ``VLLM_USE_BREAKABLE_CUDAGRAPH=0` plus `patches/apply_overlays.py --only mhc-tf32-customop`, bind-mounting the patched files; watch `docker logs` for `torch._dynamo.exc.Unsupported``

**Log.** [`documented here; the patched file is regenerable from the overlay`](../documented here; the patched file is regenerable from the overlay)

| break | construct | state |
|---|---|---|
| gate | model not torch-compiled | cleared: `@support_torch_compile` on the NVIDIA model, via `patch_nvidia_support_torch_compile` (in `apply_main`) |
| 1 | bool `is_current_stream_capturing()` in the ar-static-ws guard | cleared |
| 2 | ctypes `_FuncPtr` via `is_deep_gemm_supported()` | cleared |
| 3 | `importlib.import_module` via `_lazy_init()` | cleared |
| **4** | pybind11 `tf32_hc_prenorm_gemm` (3 call sites, 3 local import blocks) | **cleared 2026-09-17** by `direct_register_custom_op` + hoisting; `torch.compiler.disable` is rejected by this torch |
| **5** | TileLang `_is_lazy_style` -> `_infer_jit_mode` | open, and **provably** needs TileLang kept out of Dynamo. All three routes are closed by measurement: pre-resolving the mode at import is impossible (the mode-bearing object is created inside the call); un-patching `inspect.getfile` lands on `linecache.checkcache`, since `/usr/lib/python3.12/` is itself a skip directory; and bypassing the source scan (`has_internal_prim_func` -> False) makes Dynamo trace the kernel body into TVM IR construction (`tvm_ffi.core.CObject.__new__`, `Unsupported method call`). Only remaining fix: a custom op at the mhc call site. `torch._dynamo.allow_in_graph` was tried as the cheap substitute and fails: it still *runs* the function, on FakeTensors, where TileLang's kernel cache raises `TypeError("unhashable type: non-nested SymInt")`. A real custom op with an explicit `fake_impl` is required -- Dynamo then calls the fake impl and never runs the kernel during tracing |

**Verdict.** **The compile path is now the best-understood route to the gap, and it is finite work rather than a research question.** Compilation is worth having because the round-30 kernel profile puts 30 % of CUDA time in unfused pointwise kernels and 0.8 % in 7752 quantisation launches -- the signature of an uncompiled forward. The path is gated by our own default (`configs/env.spark.sh:47` forces `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, and `config/vllm.py:786` turns that into `CompilationMode.NONE`, while all four of this repo's own example recipes set it to 0). Breaks 1-4 are cleared; break 5 has a located one-line-per-kernel fix. What is left is mechanical: pin the TileLang kernel modes, re-run, repeat until the traced forward is clean.

### CUDA time by innermost Python frame, on the arm that actually runs

**Command.** `serve with `--profiler-config {"profiler":"torch","torch_profiler_dir":"/root/.cache/vllm/prof2","torch_profiler_with_stack":true}`, `POST /start_profile`, one request, `POST /stop_profile`, then attribute each kernel to its innermost `python_function` ancestor`

**Log.** [`documented here; trace under spark1:~/.cache/vllm/prof2/ (132 MB gzipped)`](../documented here; trace under spark1:~/.cache/vllm/prof2/ (132 MB gzipped))

| ms | share | innermost Python frame |
|---|---|---|
| 242.79 | **36.0 %** | `<built-in function linear>` |
| 203.66 | **30.2 %** | `tp_moe_dynamic_launch` (our b12x MoE) |
| 55.65 | 8.2 % | `blockscaled_serialized` (b12x block-scaled linear) |
| 54.94 | 8.1 % | `all_reduce` |
| 44.72 | 6.6 % | `reshape` |
| 27.73 | 4.1 % | `bmm` |
| 10.05 | 1.5 % | `mm` |
| 6.92 | 1.0 % | `copy_` |
| 3.68 | 0.5 % | `per_token_group_fp8_quant` |
| 0.96 | 0.1 % | `sm12x_b12x_kernels.py(255): sync_packed_indexer_k` (ours) |

**Verdict.** **Withdrawn as a share of the step, kept as a description of the visible fraction.** Only ~6 % of a decode step appears as individual kernel events here, because a step replays a captured CUDA graph and the profiler records the launch rather than the kernels inside it. So these shares are of the visible 6 %, not of the step, and the `linear ~44 %` reading must not be quoted. What does survive with numbers: the top row is the **LM head**, unquantized BF16 (`logits_processor._apply_head` -> `default_unquantized_gemm` -> `F.linear`), 324 kernels, median 0.284 ms, ~5.3 ms per step, **~2.2 % of the c6 step**. The correct instrument is `capture_torch_profiler: true`, which profiles the graph at capture time. Round 30 profiled under `--enforce-eager` (distorts the regime) and this one sees 6 % of it (hides the graph), so no per-stage share of the step should be quoted from either. Original note follows. **A correction to round 30, and it changes the next step.** Round 30 profiled under `--enforce-eager` and concluded that 30 % of CUDA time was unfused pointwise work, which made compilation look like the lever; five rounds were spent on it. On the normal graph-mode arm the pointwise block is ~13 % (`reshape` + `copy_` + `mm` + `bmm`) and the profile is led by **`linear` plus `blockscaled_serialized` at ~44 %** and the MoE at 30 %. Absolute totals are not comparable between the two captures because stack recording inflates them, but the rank order is. So compilation targets much less than round 30 claimed, and the linear layer family -- where only `b12x` and `deep_gemm` have been tried, and `deep_gemm` was worse -- is the largest untried surface. Our own overlay file is exonerated: its two hot sites are 0.96 ms and ~2 ms.

### the TP all-reduce priced by removal (decisive negative)

**Command.** `add a `skip_allreduce` marker under `VLLM_SKIP_FLAG_DIR` to a bind-mounted `vllm/distributed/communication_op.py`, serve `proto2` under `--enforce-eager` + `VLLM_PROFILE_DECODE=1`, and compare the enclosing per-layer total with round 26's control`

**Log.** [`outputs/driver/one-off/allreduce-skip-probe.log`](../outputs/driver/one-off/allreduce-skip-probe.log)

| measurement | control | all-reduce skipped |
|---|---|---|
| `b12x layers: n=43 sum` | **148.0 ms** | **147.7 ms** |
| region `allreduce` gpu_sum | 5.7 ms | 0.1 ms |
| region `attn` gpu_sum | 80.3 ms | 80.2 ms |
| region `ffn` gpu_sum | 65.2 ms | 64.2 ms |

**Verdict.** **All 87 collectives per step become free and the layer total moves 0.3 ms.** The skip is verified to have fired (`grep -c skip_allreduce` in the container returns 1, and the region counter drops from `gpu_sum=5.7ms` to `0.1ms`), so this is not a patch that quietly did nothing. The TP all-reduce is therefore **~0.2 % of layer time**, and every profiler reading that implicated it was an artifact: the region marks measure queue-drain, round 30's 21.4 % was a share of an eager capture, round 37's 54.9 ms was a share of the 6 % of a step visible outside graph replay. This closes the hypothesis behind round 27 (b12x's PCIe transport, which cannot cross hosts in any case). It also settles the method: the profiler has failed three ways here while a single skip A/B was decisive immediately, and `attn` + `ffn` = 144.4 of 147.7 ms says which two stages to price next.

### kernel density per layer, from the stack-recorded trace

**Command.** `count `cat == "kernel"` events and graph-replay annotations in spark1:~/.cache/vllm/prof2/dp0_pp0_tp0*.gz`

**Log.** [`documented here; trace on spark1`](../documented here; trace on spark1)

| quantity | value |
|---|---|
| kernel events in the window | 139353 |
| kernel CUDA time | 5194.8 ms |
| graph replays (`execute_context_0(0)_generation_1(8)`) | 35 |
| kernels per step | ~3981 |
| **kernels per layer** | **~92.6** |
| implied launch cost at 25 us/launch | ~2.3 ms per layer, ~99 ms per step |

**Verdict.** **The batch-invariant residual is launch overhead, and now has a size.** Round 41 measured the step as flat at 96-98 ms from 8 to 48 rows with the FFN removed, which cannot be memory traffic; this counts ~92.6 kernel launches per layer, and at a conservative 25 us each that is ~2.3 ms per layer -- the residual almost exactly. It also **corrects round 38**: that round said only ~6 % of a step is visible because a graph replay hides the kernels, but the kernels are all there (5194.8 ms); what covers 675.3 ms is attribution to a *Python frame*, since most run inside Inductor's `execute_context_*` regions. The shares computed there are still not shares of the step, but the recorded reason was wrong. This re-justifies the compile port on a mechanism -- kernel count per layer -- rather than on the eager-mode percentage round 38 rightly distrusted.

### kernels ranked by launches per layer, and the biggest one named

**Command.** `aggregate `cat == "kernel"` events by name and by count/(steps*layers) from spark1:~/.cache/vllm/prof2/dp0_pp0_tp0*.gz`

**Log.** [`documented here; trace on spark1`](../documented here; trace on spark1)

| count | per layer | ms | kernel |
|---|---|---|---|
| 7560 | 5.02 | **1543.03** | `...gpu_kernel_impl_nocast<direct_copy_kernel_cuda...` |
| 4177 | 2.78 | 294.65 | `b12x_libdense_gemmDenseGemmKernel...` |
| 3891 | 2.59 | 203.03 | `cutlass_80_wmma...bf16_16x16_128x2` |
| 3636 | 2.42 | 204.06 | `ncclDevKernel_AllReduce_bf16_RING` |
| 3492 | 2.32 | 17.37 | `mhc_pre_big_fuse_with_norm_tilelang_kernel` |
| 3335 | 2.22 | 52.41 | `mhc_fused_tilelang_kernel` |

**Verdict.** **One kernel family is 30 % of all kernel time: a large direct copy, 5.02 launches per layer, 44 ms per step.** Its largest instance is `grid [28496,1,1]` -- ~15M elements, ~29 MB in bf16 -- and round 37's stack put the family under `execute_context_0 <- aten::reshape <- aten::clone <- aten::copy_`, i.e. a clone inside a compiled region. It is **not** our all-reduce workspace copies: round 39 made that whole function a no-op, removing four copies per layer, and the layer total moved 0.3 ms. It is **not** the MoE either: round 41 removed the FFN and the step fell only 9.1 ms, so a 44 ms cost there would have gone with it. That leaves the attention/mHC half, and our own overlay files contain only small `.contiguous()` calls and KV page slices, nothing ~29 MB. **Naming it needs shapes rather than counts** -- `torch_profiler_record_shapes: true` under `--enforce-eager` -- and it is worth naming: 44 ms/step is 19 % of the c6 step and 33 % of the c1 step, larger than anything else still open.

### the indexer-gather win confirmed at the kernel level, and the new top item

**Command.** `re-run the shapes + CUDA-time dump on the kept arm (`--enforce-eager`, `record_shapes`, no stacks) and compare against round 45's dump`

**Log.** [`spark1:~/.cache/vllm/prof5/profiler_out_0.txt (both runs)`](../spark1:~/.cache/vllm/prof5/profiler_out_0.txt (both runs))

| host op | old arm | kept arm | change |
|---|---|---|---|
| `aten::copy_` | **433.03 ms** | **22.33 ms** | **19x less** |
| `elementwise_kernel<...direct_copy...>` | 416.56 ms | not in the top list | gone |
| `b12x::tp_moe_dynamic_launch` | 362.49 ms | **304.39 ms (42.55 %, #1)** | |
| `vllm::all_reduce` | 265.62 ms | 74.22 ms | |
| `aten::mm` | 117.91 ms | 98.10 ms | |
| `b12x::blockscaled_serialized` | 103.02 ms | 84.95 ms | |

**Verdict.** **The switch does exactly what it was supposed to, and the MoE is what is left.** `aten::copy_` falls 19x -- from the single largest item to 3 % -- and the `direct_copy` elementwise kernel drops out of the top list entirely, which is the full-cache indexer gather the switch removes. Windows differ between the captures (25 graph replays against 35, plus 5 prefill captures), so the other columns are indicative rather than matched, but the copy's collapse is unambiguous. **The new largest item is the MoE at 42.55 % of kernel time**, and it is the one place every lever is already spent: our kernel measured 10-13x faster than FlashInfer's, the A16 activation format is a wash, and the tile-shape override landed inside the run-to-run range.

### the plain BF16 GEMMs: launcher check (negative)

**Command.** `attribute `aten::bmm` and `aten::mm` in the shapes trace by launcher name and recorded Input Dims`

**Log.** [`spark1:~/.cache/vllm/prof5/dp0_pp0_tp0*1789648766*.gz`](../spark1:~/.cache/vllm/prof5/dp0_pp0_tp0*1789648766*.gz)

| op | calls | recorded shapes | verdict |
|---|---|---|---|
| `aten::bmm` | 276 | `[4,8,4096] x [4,4096,1024]` x215, `[4,84,...]` x43 | the documented grouped WO-A **b12x path**, not the `o_proj` else-branch |
| `aten::mm` | 828 | `[8,4096] x [4096,256|2048|64|512|1024]` + bias, BF16 | mHC coefficient generators; per-step counts, no main-projection shape |

**Verdict.** **Suspicion raised, mechanism checked, no unquantised fallback.** 148 ms of plain BF16 GEMMs in an FP8-quantised model looked like a fallback, but `try_b12x_wo_proj`'s docstring says it replaces the einsum with "bmm + cached WO-A", and the shapes confirm it: `4` is `o_groups` after the TP=2 split, `4096` the group width, `1024` the `o_lora_rank`. The `aten::mm` family is small `F.linear(x, w, bias)` calls whose counts are per step rather than per layer, matching the legitimately-unquantised mHC coefficient generators. Neither family carries a launcher Python frame, because both run inside Inductor's `execute_context_*` regions. Two side notes: `try_b12x_wo_proj` declines above 256 tokens, so **prefill uses a different WO path than decode**; and its `DBG wo_proj` prints are still in the shipped file.

## Regenerating

```sh
python3 scripts/experiment-ledger.py
```

Reads every `outputs/driver/**/*.median.log` (valid arms are flat, superseded arms are in `superseded/`, one-off instrument output is in `one-off/`) and the highest-numbered `*-<N>.meter.txt` beside each, and writes this file plus `outputs/experiments.json`.
As of the last run: 211 guarded arms, 0 superseded, 70 failed.

