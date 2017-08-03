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

#ifndef STREAM_BASED_AL_EXPERIMENTER_H
#define STREAM_BASED_AL_EXPERIMENTER_H

/* Boost */
#include <boost/progress.hpp>
#include <boost/timer.hpp>
/* Mondrian */
#include "stream_based_al_forest.h"
#include "stream_based_al_utilities.h"
#include "stream_based_al_hyperparameters.h"
#include "stream_based_al_data.h"


/*---------------------------------------------------------------------------*/
/**
 * Experimenter class to train and test a Mondrian Forest
 */
class Experimenter {
  public:
    /*
     * Construct experimenter
     */
    Experimenter();
    Experimenter(const bool confidence = false);
    //Experimenter(MondrianForest* mf, Dataset& dataset, Hyperparameters& hp);
    ~Experimenter();

    /**
    * Function trains a mondrian forest
    * 
    * Input parameter:
    *
    * @param mf        : A Mondrian forest
    * @param dataset   : Trainings dataset
    * @param hp        : Hyperparameters
    *
    */
    void train(MondrianForest* mf, DataSet& dataset, Hyperparameters& hp);

    /**
    * Function trains a mondrian forest in an active learning setting
    *
    * Input parameter:
    *
    * @param mf        : A Mondrian forest
    * @param dataset   : Trainings dataset
    * @param hp        : Hyperparameters
    *
    */
    void train_active(MondrianForest* mf, DataSet& dataset, Hyperparameters& hp);

    /**
     * Function tests/evaluates a mondrian forest
     * 
     * Input parameter:
     *
     * @param mf        : A Mondrian forest
     * @param dataset   : Testing dataset
     * @param hp        : Hyperparameters
     *
     * Output: accuracy
     *
     */
    double test(MondrianForest* mf, DataSet& dataset, Hyperparameters& hp);

    /**
     * Return training time
     */
    double get_training_time();
    /**
     * Return testing time
     */
    double get_testing_time();
    /**
     * Return accuracy value
     */
    double get_accuracy();
    /** 
     * Return detailed result
     */
    Result get_detailed_result();

  private:
    const bool conf_value_;  /**< Set option: returns confidence value of each
                         prediction */
    Result* pResult_;  /** Saves all results in a defined structure */

    /**
     * Evaluate test results
     *
     * Output parameter:
     *  - returns accuracy
     */
    double evaluate_results(DataSet& dataset_test);

};

#endif // STREAM_BASED_AL_EXPERIMENTER_H
