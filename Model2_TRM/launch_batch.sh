run=bash

# for lang in mlt swa crh; do
for lang in mwf izh mlt; do
    
    steps=4000
    #steps=$((steps+7000))
    for vers in _d1k_o1_c_lcs-3 _d1k_o1_f_lcs-3; do
    
        # regular data
        echo trm $lang $vers
        bash example/sigmorphon2020-shared-tasks/task0-trm_batch.sh $lang$vers $steps
        #steps=$((steps+7000))
        # echo mono $lang
        # bash example/sigmorphon2020-shared-tasks/task0-mono.sh $lang

        # # augmented data
        # echo trm hall $lang
        # $run example/sigmorphon2020-shared-tasks/task0-hall-trm.sh $lang
        # echo mono hall $lang
        # $run example/sigmorphon2020-shared-tasks/task0-hall-mono.sh $lang
    done
done

# for lang in afro-asiatic austronesian dravidian germanic indo-aryan iranian niger-congo oto-manguean romance turkic uralic; do
#     # regular data
#     echo trm $lang
#     $run example/sigmorphon2020-shared-tasks/task0-trm.sh $lang
#     echo mono $lang
#     $run example/sigmorphon2020-shared-tasks/task0-mono.sh $lang

#     # augmented data
#     echo trm hall $lang
#     $run example/sigmorphon2020-shared-tasks/task0-hall-trm.sh $lang
#     echo mono hall $lang
#     $run example/sigmorphon2020-shared-tasks/task0-hall-mono.sh $lang
# done
