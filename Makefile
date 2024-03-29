DIRS = $(shell ls -d */ | grep -v ptmpls)

.PHONY: all
all: hardware.txt list.txt environment.yml recursive

hardware.txt:
	echo '$ lscpu:' > hardware.txt
	lscpu >> hardware.txt
	echo >> hardware.txt
	echo '$ lsmem:' >> hardware.txt
	lsmem >> hardware.txt
	echo >> hardware.txt
	echo '$ uname -a:' >> hardware.txt
	uname -a | awk '$$2="********"' >> hardware.txt
	echo >> hardware.txt
	echo '$ gcc --version' >> hardware.txt
	gcc --version >> hardware.txt
	echo >> hardware.txt
	echo '$ icc --version' >> hardware.txt
	icc --version >> hardware.txt

environment.yml:
	conda env export | grep -v prefix > environment.yml

list.txt:
	ls */*.png > list.txt

.PHONY: recursive
recursive: template
	for DIR in $(DIRS);\
	do\
		$(MAKE) -C $${DIR};\
	done

.PHONY: template
template:
	$(MAKE) -C ptmpls

.PHONY: environment
environment: environment.yml
	conda env create -f environment.yml
