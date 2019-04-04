PGFOBJS = Figure1.pgf Figure2.pgf Figure3.pgf Figure4.pgf Figure5.pgf Figure6.pgf Figure7.pgf Figure8.pgf Figure9.pgf
RUNOBJS = Problem.run

CDEPS = quad/mid.c quad/trap.c quad/simp.c quad/romb.c quad/lagu.c quad/lege.c quad/wrappers.c quad/quad.h

ifeq ($(CC),cc)
ifneq (,$(shell which icc))
		CC = icc
endif
endif
PYCC = $(CC) -pthread
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

quad.so: setup.py setup.cfg $(CDEPS)
	CC="$(PYCC)" LDSHARED="$(PYLD)" python setup.py build_ext --inplace
	mv `ls quad.*.so` quad.so

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py quad.so
	python $<
	touch $@

.PHONY: run
run: $(RUNOBJS)
