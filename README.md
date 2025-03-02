# template-matching-play

Look for an image (template) inside another image, by [normalized
cross-correlation](https://scribblethink.org/Work/nvisionInterface/nip.html)

Goal to use no dependencies (no OpenCV, no Python, no NumPy, etc),
just pure C and maybe Apple math libraries.

See my [!!con 2022 talk about this](https://www.youtube.com/watch?v=fH6yjw_PlEU).

To test Python version:

```
$ python3 template-matching-play.py
```

To test 'Objective-C++' (mostly pure C + Apple Accelerate, tbh) version:

```
$ make && ./template-matching-play-apple-vdsp
```

Part of work on ScreenMatcher (which is not done/released yet, but here's a [demo
video](https://www.youtube.com/watch?v=TDhJwt7fLLs) & [presentation at
Future of Text](https://www.youtube.com/watch?v=y04EqH6x_zk) &
[podcast](https://museapp.com/podcast/73-folk-practices/)). ScreenMatcher
has a somewhat more up-to-date / bug-fixed version that I haven't
backported to this; there may be some bugs in this one.

Apache 2.0 license.
