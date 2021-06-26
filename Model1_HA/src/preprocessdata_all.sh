for i in '.dev' '.trn' '.tst';
do

#python3 canonicalize.py ../data/fin/fin"$i" > ../data/fin/fin"$i".pos
#cat ../data/fin/fin"$i".pos | awk -F"\t" 'BEGIN {OFS="\t"};{gsub(/\r/,"",$3); print $1,$3,$2}' > ../data/fin/fin"$i".r
#rm ../data/fin/fin"$i".pos

#python3 canonicalize.py ../data/mlt/mlt"$i" > ../data/mlt/mlt"$i".pos
#cat ../data/mlt/mlt"$i".pos | awk -F"\t" 'BEGIN {OFS="\t"};{gsub(/\r/,"",$3); print $1,$3,$2}' > ../data/mlt/mlt"$i".r
#rm ../data/mlt/mlt"$i".pos


#python3 canonicalize.py ../data/swa/swa"$i" > ../data/swa/swa"$i".pos
#cat ../data/swa/swa"$i".pos | awk -F"\t" 'BEGIN {OFS="\t"};{gsub(/\r/,"",$3); print $1,$3,$2}' > ../data/swa/swa"$i".r
#rm ../data/swa/swa"$i".pos



python3 canonicalize.py ../data/$1/$1"$i" > ../data/$1/$1"$i".pos
cat ../data/$1/$1"$i".pos | awk -F"\t" 'BEGIN {OFS="\t"};{gsub(/\r/,"",$3); print $1,$3,$2}' > ../data/$1/$1"$i".r
rm ../data/$1/$1"$i".pos







done
