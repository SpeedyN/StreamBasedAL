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

#ifndef STREAM_BASED_AL_HYPERPARAMETERS_H_
#define STREAM_BASED_AL_HYPERPARAMETERS_H_

#include <string>
using namespace std;

/*
 * Hyperparameters of Mondrian Forest
 */
class Hyperparameters {
    public:
        Hyperparameters(const string& conf_file);

        /* Data files */
        string train_data_;  /**< Data with training data */
        string train_labels_;  /**< Data with labels of training data */
        string test_data_;  /**< Data with testing data */
        string test_labels_;  /**< Data with labels of testing data */

        /* Parameters how data should be load */
        bool random_;  /**< Shuffle data randomly */
        bool iterative_;  /**< Jumps to define line in file and loads only
                            data of that line iteratively*/
        bool sort_data_;  /**< Sorts data after classes */
        bool training_data_in_diff_files_;  /**< Training data in different
                                              files */

        /* Parameters for Mondrian forest */
        int num_trees_;  /**< Number of trees of Mondrian forest */
        float init_budget_;  /**< Initial budget of a Mondrian tree */
        float discount_factor_;  /**< Discount parameter of a Mondrian tree */
        float decision_prior_hyperparam_; /**< Hyperparameter of the beta prior of each node */
        bool debug_;  /**< Debug mode */
        int max_samples_in_one_node_;  /**< Splits a node if this number
                                         is reached */
        bool print_properties_;  /**< Print properties of a Mondrian Forest */

        /* Parameters for training */
        unsigned int number_of_samples_for_training_;  /**< Number of 
                                                           training samples */
        /**
         * Set option active learning:
         *  - 0 = no active learning
         *  - 1 = active learning: use samples that are lower than value 
         *                         "active_confidence_value_"
         *  - 2 = active learning: use only "active_buffer_percentage" samples
         *                         of the training set for training
         */
        int active_learning_;  /**< Set active learning procedure */
        int active_number_init_set_;  /**< Number of training samples for the
                                       initial training round */
        int active_batch_size_;  /**< Batch size of a training round in the case
                                  of active learning */
        /**< Sets option to train classifier only with the samples
          with a low confidence (percentage) */
        bool active_buffer_lowest_confidence_;
        int active_buffer_size_;  /**< Percentage of samples that have
                                         a low confidence and are used to train
                                         the classifier */
        float active_confidence_value_;

};

#endif /* STREAM_BASED_AL_HYPERPARAMETERS_H_ */
