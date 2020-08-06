for fN in $(ls|grep "test_*")
do
	python $fN $1
done
