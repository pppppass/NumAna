PGFOBJS = Figure1.pgf Figure2.pgf Figure3.pgf Figure4.pgf Figure5.pgf Figure6.pgf Figure7.pgf Figure8.pgf Figure9.pgf
TBLOBJS = Table1.tbl Table2.tbl
RUNOBJS = Problem.run

CDEPS = intp/newt.c intp/lagr.c intp/lin.c intp/cub.c intp/spl_cub.c intp/utils.c intp/wrappers.c intp/intp.h

ifeq ($(CC),cc)
ifneq (,$(shell which icc))
		CC = icc
endif
endif
PYCC = $(CC) -pthread
PYLD = $(CC) -pthread -shared

.PHONY: all
all: Report.pdf

Report.pdf: Report.tex $(PGFOBJS) $(TBLOBJS) | pnote.cls pdef.sty Report.aux
	lualatex Report.tex

Report.aux: Report.tex | pnote.cls pdef.sty
	lualatex Report.tex

pnote.cls pdef.sty:
	ln -s ptmpls/$@ .

$(TBLOBJS): $(wildcard $(RUNOBJS))
	touch $@

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

intp.so: setup.py setup.cfg $(CDEPS)
	CC="$(PYCC)" LDSHARED="$(PYLD)" python setup.py build_ext --inplace
	mv `ls intp.*.so` intp.so

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py op.so $(PYDEPS)
	python $<
	touch $@

.PHONY: run
run: $(RUNOBJS)
