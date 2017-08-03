// -*- C++ -*-
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 or the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2016
 * Dep. Of Computer Science
 * Technical University of Munich (TUM)
 *
 */
#include <sys/time.h>
#include <vector>
#include <string.h>
#include <stdio.h>
#include <sstream>
/* Armadillo */
#include <armadillo>
/* Boost */
#include <boost/progress.hpp>
#include <boost/timer.hpp>
/* Mondrian libraries */
#include "stream_based_al_forest.h"
#include "stream_based_al_data.h"
#include "stream_based_al_utilities.h"
#include "stream_based_al_hyperparameters.h"
#include "stream_based_al_experimenter.h"

/*
 * Help function
 */
void help() {
    cout << endl;
    cout << "Help function of StreamBased_AL: " << endl;
    cout << "Input arguments: " << endl;
    cout << "\t -h | -- help: \t will display help message." << endl;
    cout << "\t -c : \t\t path to the config file." << endl << endl;
    cout << "\t --train : \t Train the classifier." << endl;
    cout << "\t --test  : \t Test the classifier." << endl;
    cout << "\t --confidence: \t Calculates a confidence value for each prediction \n \t\t\t (works but will not be saved in some file)" << endl;
    cout << "\tExamples:" << endl;
    cout << "\t ./StreamBasedAL_MF -c conf/stream_based_al.conf --train --test" << endl;
}

int main(int argc, char *argv[]) {
    cout << endl;
    cout << "################" << endl;
    cout << "StreamBased_AL: " << endl;
    cout << "################" << endl;
    cout << endl;
    /* Program parameters */
    bool training = false, testing = false, conf_value = false;
/*---------------------------------------------------------------------------*/
    /*
     * Reading input parameters
     * ------------------------
     */
    /* Check if input arguments are specified */
    if(argc == 1) {
        cout << "\tNo input argument specified: aborting..." << endl;
        help();
        exit(EXIT_SUCCESS);
    }
    int input_count = 1;
    string conf_file_name;
    // Parsing command line
    while (input_count < argc) {
        if (!strcmp(argv[input_count], "-h") || !strcmp(argv[input_count], 
                    "--help")) {
            help();
            return EXIT_SUCCESS;
        } else if (!strcmp(argv[input_count], "-c")) {
            conf_file_name = argv[++input_count];
        } else if (!strcmp(argv[input_count], "--train")) {
            training = true;
        } else if (!strcmp(argv[input_count], "--test")) {
            testing = true;
        } else if (!strcmp(argv[input_count], "--confidence")) {
            conf_value = true;
        } else {
            cout << "\tUnknown input argument: " << argv[input_count];
            cout << ", please try --help for more information." << endl;
            help();
            exit(EXIT_FAILURE);
        }
        input_count++;
    }
    if (conf_file_name.length() < 1) {
        cout << "[ERROR] - No config file selected ... " << endl;
        help();
        exit(EXIT_FAILURE);
    }
   
/*---------------------------------------------------------------------------*/
    /*
     * Loading training data and get properties of data set
     * ----------------------------------------------------
     */
    /* Load hyperparameters of Mondrian forest */
    Hyperparameters hp(conf_file_name);

    cout << endl;
    cout << "------------------" << endl;
    cout << "Loading files  ..." << endl;
    cout << "------------------" << endl;
    /* Load training and testing data */
    DataSet dataset_train(hp.random_, hp.sort_data_, hp.iterative_);
    DataSet dataset_test;
    dataset_train.load(hp.train_data_, hp.train_labels_);
    dataset_test.load(hp.test_data_, hp.test_labels_);
    /* Set feature dimension */
    int feat_dim = dataset_train.feature_dim_;

    /*
     * Set settings of Mondrian forest
     * ----------------------------------------------------
     */
    mondrian_settings* settings = new mondrian_settings;
    settings->num_trees = hp.num_trees_; 
    settings->discount_factor = hp.discount_factor_;
    settings->discount_param = settings->discount_factor * float(feat_dim);
    settings->debug = hp.debug_;
    settings->max_samples_in_one_node = hp.max_samples_in_one_node_;

/*---------------------------------------------------------------------------*/
    /* Initialize Mondrian forest */
    MondrianForest* forest = new MondrianForest(*settings, feat_dim);
    
    /* Initialize experimenter class */
    Experimenter experimenter(conf_value);
    
    if (training) {
      /* Option between active learning and without */
      if (hp.active_learning_ > 0)
        experimenter.train_active(forest, dataset_train, hp);
      else
        experimenter.train(forest, dataset_train, hp);
    }
    if (testing) {
      double accuracy = experimenter.test(forest, dataset_test, hp);

      cout << endl;
      cout << "------------------" << endl;
      cout << "Properties:       " << endl;
      cout << "------------------" << endl;
      cout << "Accuracy: \t" << accuracy << endl;
      cout << endl;
      Result result = experimenter.get_detailed_result();
      cout << "Samples used for training: "
        << result.samples_used_for_training_ << endl;
      cout << endl;

    }

/*---------------------------------------------------------------------------*/
    /*
     * Free Space
     */
    delete forest;
    delete settings;

    return 0;
}
