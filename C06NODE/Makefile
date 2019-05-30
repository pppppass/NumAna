PGFOBJS = Figure01.pgf Figure02.pgf Figure03.pgf Figure04.pgf Figure05.pgf Figure06.pgf Figure07.pgf Figure08.pgf Figure09.pgf Figure10.pgf Figure11.pgf Figure12.pgf Figure13.pgf Figure14.pgf Figure15.pgf Figure16.pgf Figure17.pgf Figure18.pgf Figure19.pgf Figure20.pgf Figure21.pgf Figure22.pgf Figure23.pgf Figure24.pgf Figure25.pgf Figure26.pgf Figure27.pgf Figure28.pgf Figure29.pgf Figure30.pgf Figure31.pgf Figure32.pgf Figure33.pgf Figure34.pgf Figure35.pgf Figure36.pgf Figure37.pgf Figure38.pgf Figure39.pgf Figure40.pgf Figure41.pgf Figure42.pgf Figure43.pgf Figure44.pgf Figure45.pgf Figure46.pgf Figure47.pgf Figure48.pgf
RUNOBJS = Problem.run

CDEPS = intg/eval.c intg/ode1.c intg/ode4.c intg/wrappers.c intg/intg.h

ifeq ($(CC),cc)
ifneq (,$(shell which icc))
		CC = icc
endif
endif
PYCC = $(CC) -pthread -Werror -Wno-error=strict-prototypes -Wno-error=unused-function
PYLD = $(CC) -pthread -shared

.PHONY: all
all: Report.pdf

Report.pdf: Report.tex $(PGFOBJS) | pnote.cls pdef.sty Report.aux
	lualatex Report.tex

Report.aux: Report.tex | pnote.cls pdef.sty
	lualatex Report.tex

pnote.cls pdef.sty:
	ln -s ../ptmpls/$@ .

$(PGFOBJS): $(wildcard $(RUNOBJS))
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

intg.so: setup.py setup.cfg $(CDEPS)
	CC="$(PYCC)" LDSHARED="$(PYLD)" python setup.py build_ext --inplace
	mv `ls intg.*.so` intg.so

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py fft.so
	python $<
	touch $@

.PHONY: run
run: $(RUNOBJS)
