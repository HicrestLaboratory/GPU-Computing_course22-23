#!/bin/bash

if [[ $# < 2 ]] || [[ ${2: -1} != '/' ]]
then
    echo "usage: 2remote.sh remote_user@remote_id destination_folder_on_remote [ssh_key]"
    echo "NOTE: destination_folder_on_remote must end with '/'"
else

    remote=$1
    DEST_FOLDER=$2
    test=${DEST_FOLDER: -1}
    if [[ $test != '/' ]]
    then
        echo "non /"
    else
        echo "è /"
    fi

    if [[ $# > 2 ]]
    then
        sshk=$3
    fi

    CURRENTEPOCTIME=`date +"%s"`
    mkdir ~/tmp$CURRENTEPOCTIME

    folder_contenent() {
        for FILE in $1/*
        do
            if [[ -d $FILE ]]
            then
    #             echo "Directory: " $FILE
    #             echo "Create folder: " $DEST_FOLDER$FILE
                echo "Creating $FILE on tmp"
                mkdir ~/tmp$CURRENTEPOCTIME/$DEST_FOLDER$FILE
                folder_contenent $FILE
            else
    #             echo "File:      " $FILE
                cp $FILE ~/tmp$CURRENTEPOCTIME/$DEST_FOLDER$FILE
            fi
        done
    #     echo "  ------  "
    }

    echo "Creating $DEST_FOLDER on tmp"
    mkdir ~/tmp$CURRENTEPOCTIME/$DEST_FOLDER
    for FILE in *
    do
        if [[ -d $FILE ]]
        then
    #         echo "Directory: " $FILE
    #         echo "Create folder: " $DEST_FOLDER$FILE
            echo "Creating $FILE on tmp"
            mkdir ~/tmp$CURRENTEPOCTIME/$DEST_FOLDER$FILE
            folder_contenent $FILE
        else
    #         echo "File:      " $FILE
            cp $FILE ~/tmp$CURRENTEPOCTIME/$DEST_FOLDER$FILE
        fi
    done


    if [[ $# > 2 ]]
    then
        ssh -i $sshk $remote "rm -rf $DEST_FOLDER; exit;"
        scp -i $sshk -r ~/tmp$CURRENTEPOCTIME/$DEST_FOLDER $remote:$DEST_FOLDER
    else
        ssh $remote "rm -rf $DEST_FOLDER; exit;"
        scp -r ~/tmp$CURRENTEPOCTIME/$DEST_FOLDER $remote:$DEST_FOLDER
    fi
    rm -r -f ~/tmp$CURRENTEPOCTIME

fi
