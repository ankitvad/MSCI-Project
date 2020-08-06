read -a list -p "Please give the list of epoch to be tested: "

for name in "${list[@]}"; do
	#echo "$name"
	python train_continue.py test "$name"
done