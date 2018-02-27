SVG_FILES := $(wildcard *.svg)
PDF_FILES := $(patsubst %.svg,%.pdf,$(SVG_FILES))

all: $(PDF_FILES)

%.pdf : %.svg
	inkscape --export-pdf=$@ $< 2> /dev/null
