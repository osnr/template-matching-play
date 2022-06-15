template-matching-play-apple-vdsp: template-matching-play-apple-vdsp.mm
	c++ -lobjc -framework Cocoa -framework Accelerate -O2 -g -std=c++11 `pkg-config --cflags --libs opencv4` -o $@ $<

apple-vdsp-prof: template-matching-play-apple-vdsp
	killall -9 $< ; ./$< &
	sample $< 2 -wait -e
