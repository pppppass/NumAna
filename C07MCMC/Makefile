PGFOBJS = Figure01.pgf Figure02.pgf Figure03.pgf Figure04.pgf Figure05.pgf Figure06.pgf Figure07.pgf Figure08.pgf Figure09.pgf Figure10.pgf Figure11.pgf Figure12.pgf Figure13.pgf Figure14.pgf Figure15.pgf Figure16.pgf Figure17.pgf Figure18.pgf Figure19.pgf Figure20.pgf Figure21.pgf
RUNOBJS = Problem1.run Problem2.run Problem3.run Problem4.run

CDEPS = samp/sing_2d.c samp/metr_2d.c samp/kin_2d.c samp/kin_3d.c samp/utils.c samp/wrappers.c samp/samp.h

ifeq ($(CC),cc)
ifneq (,$(shell which icc))
		CC = icc
endif
endif
PYCC = $(CC) -pthread# -Werror -Wno-error=strict-prototypes -Wno-error=unused-function
PYLD = $(CC) -pthread -shared

.PHONY: all
all: Report.pdf

Report.pdf: Report.tex $(PGFOBJS) | pnote.cls pdef.sty Report.aux
	lualatex Report.tex

Report.aux: Report.tex | pnote.cls pdef.sty
	lualatex Report.tex

pnote.cls pdef.sty:
	ln -s ../ptmpls/$@ .

$(PGFOBJS): Plot.py $(wildcard $(RUNOBJS))
	python Plot.py
	sed -i 's/−/-/g' $@
	touch $@

setup.cfg:
	if [ -z ${CONDA_PREFIX} ];\
	then\
		echo "Anaconda not found";\
		exit 1;\
	fi
	echo "[build_ext]" > setup.cfg
	echo "include_dirs=${CONDA_PREFIX}/include" >> setup.cfg
	echo "library_dirs=${CONDA_PREFIX}/lib" >> setup.cfg

samp.so: setup.py setup.cfg $(CDEPS)
	CC="$(PYCC)" LDSHARED="$(PYLD)" python setup.py build_ext --inplace
	mv `ls samp.*.so` samp.so

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py samp.so
	python $<
	touch $@

.PHONY: run
run: $(RUNOBJS)
