run=bash

# for lang in mlt swa crh; do
for lang in $1; do
    # regular data
    echo trm $lang
    bash example/sigmorphon2020-shared-tasks/task0-trm.sh $lang $2
    # echo mono $lang
    # bash example/sigmorphon2020-shared-tasks/task0-mono.sh $lang

    # # augmented data
    # echo trm hall $lang
    # $run example/sigmorphon2020-shared-tasks/task0-hall-trm.sh $lang
    # echo mono hall $lang
    # $run example/sigmorphon2020-shared-tasks/task0-hall-mono.sh $lang
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
