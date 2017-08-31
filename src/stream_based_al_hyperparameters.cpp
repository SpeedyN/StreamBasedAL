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

#include <iostream>
#include <libconfig.h++>
#include "stream_based_al_hyperparameters.h"

using namespace libconfig;

Hyperparameters::Hyperparameters(const string& conf_file) {
    cout << "Loading config file: " << conf_file << " ..." << endl;

    Config config_file;
    config_file.readFile(conf_file.c_str());
    
    /* General */
    user_seed_config_ = static_cast<unsigned int> (config_file.lookup("General.seed"));

    /* Load data files */
    train_data_ = (const char *) config_file.lookup("Data.train_data");
    train_labels_ = (const char *) config_file.lookup("Data.train_labels");
    test_data_ = (const char *) config_file.lookup("Data.test_data");
    test_labels_ = (const char *) config_file.lookup("Data.test_labels");
    
    /* Parameter how data should be loaded */
    random_ = (bool)config_file.lookup("Load_data.random");
    iterative_ = (bool)config_file.lookup("Load_data.iterative");
    sort_data_ = (bool)config_file.lookup("Load_data.sort_data");
    training_data_in_diff_files_ = (bool)config_file.lookup(
        "Load_data.training_data_in_diff_files");

    /* Parameter for Mondrian forest */
    num_trees_ = config_file.lookup("Mondrian.num_trees");
    init_budget_ = config_file.lookup("Mondrian.init_budget");
    discount_factor_ = config_file.lookup("Mondrian.discount_factor");
    decision_prior_hyperparam_ = config_file.lookup("Mondrian.decision_prior_hyperparam");
    debug_ = (bool)config_file.lookup("Mondrian.debug"); 
    max_samples_in_one_node_ = config_file.lookup(
        "Mondrian.max_samples_in_one_node");
    confidence_measure_ = (int) config_file.lookup("Mondrian.confidence_measure");
    print_properties_ = (bool)config_file.lookup("Mondrian.print_properties");

    /* Parameters for training */
    number_of_samples_for_training_ = config_file.lookup(
        "Training.number_of_samples_for_training");
    active_learning_ = config_file.lookup("Training.active_learning");
    active_number_init_set_ = config_file.lookup(
        "Training.active_number_init_set");
    active_batch_size_ = config_file.lookup(
        "Training.active_batch_size");
    active_buffer_lowest_confidence_ = (bool)config_file.lookup(
        "Training.active_buffer_lowest_confidence");
    active_buffer_size_ = config_file.lookup(
        "Training.active_buffer_size");
    active_confidence_value_ = config_file.lookup(
        "Training.active_confidence_value");
}
