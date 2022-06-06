template-matching-play-apple-vdsp: template-matching-play-apple-vdsp.mm
	c++ -lobjc -framework Cocoa -framework Accelerate -O2 -std=c++11 `pkg-config --cflags --libs opencv4` -o $@ $<
