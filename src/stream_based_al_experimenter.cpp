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

#include "stream_based_al_experimenter.h"


/*
 * Construct experimenter
 */
Experimenter::Experimenter() :
  conf_value_(false) {
  pResult_ = new Result();
}
Experimenter::Experimenter(const bool confidence) :
  conf_value_(confidence) {
  pResult_ = new Result();
}
Experimenter::~Experimenter() {
  delete pResult_;
}
/**
* Function trains a mondrian forest
*/ 
void Experimenter::train(MondrianForest* mf, DataSet& dataset,
    Hyperparameters& hp) {

  /* Set number of training samples */
  unsigned int number_training_samples = 0;
  if (hp.number_of_samples_for_training_ == 0)
    number_training_samples = dataset.num_samples_;
  else
    number_training_samples = hp.number_of_samples_for_training_;

  cout << endl;
  cout << "------------------" << endl;
  cout << "Start training ..." << endl;
  cout << "------------------" << endl;

  /* Check if test file exists */
  if (dataset.num_samples_ < 1) {
      cout << "[ERROR] - There does not exist a training dataset" << endl;
      exit(EXIT_FAILURE);
  }
  /* Initialize progress bar */
  unsigned int expected_count= dataset.num_samples_;
  /* Display training progress */
  boost::progress_display show_progress( expected_count );

  /* Initialize stop time for training */
  timeval startTime;
  gettimeofday(&startTime, NULL);

  /*---------------------------------------------------------------------*/
  /* Go through complete training set */
  int long i_samp = 0; 
  //for (; i_samp < hp.number_of_samples_for_training_; i_samp++) {
  for (; i_samp < number_training_samples; i_samp++) {
      Sample sample = dataset.get_next_sample();
      mf->update(sample);
      pResult_->samples_used_for_training_++;
      /* Show progress */
      ++show_progress;
  }

  /*---------------------------------------------------------------------*/
  cout << endl;
  cout << " ... finished training after: ";
  timeval endTime;
  gettimeofday(&endTime, NULL);
  float tmp_training_time = (endTime.tv_sec - startTime.tv_sec + 
          (endTime.tv_usec - startTime.tv_usec) / 1e6);
  cout << tmp_training_time << " seconds." << endl;

  pResult_->training_time_ += tmp_training_time;

}


/**
* Function trains a mondrian forest in an active learning setting
*/
void Experimenter::train_active(MondrianForest* mf, DataSet& dataset,
    Hyperparameters& hp) {
  
  /* Set number of training samples */
  unsigned int number_training_samples = 0;
  if (hp.number_of_samples_for_training_ == 0)
    number_training_samples = (int) dataset.num_samples_;
  else
    number_training_samples = hp.number_of_samples_for_training_;

  cout << endl;
  cout << "-------------------------------------" << endl;
  cout << "Start training (active learning " << 
    hp.active_learning_ << ")..." << endl;
  cout << "-------------------------------------" << endl;
  
  /* Check if test file exists */
  if (dataset.num_samples_ < 1) {
      cout << "[ERROR] - There does not exist a training dataset" << endl;
      exit(EXIT_FAILURE);
  }
  /* Initialize progress bar */
  unsigned int expected_count= dataset.num_samples_;
  /* Display training progress */
  boost::progress_display show_progress( expected_count );

  /* Initialize stop time for training */
  timeval startTime;
  gettimeofday(&startTime, NULL);

  /* Variables of active learning */
  vector<float> active_conf_values;

  /*---------------------------------------------------------------------*/
  /* Go through complete training set */
  
  /**
   * Options active learning:
   *  - 1 = active learning: updates mf with samples that are less than
   *                         "active_confidence_value_"
   *  - 2 = active learning: use only "active_buffer_percentage" samples
   *                         of the training set to update mf
   */
  if (hp.active_learning_ == 1) {
    for (int long i_samp = 0; i_samp < number_training_samples ; i_samp++) {
        Sample sample = dataset.get_next_sample();

        if (i_samp < hp.active_number_init_set_) {
            mf->update(sample);
            pResult_ -> samples_used_for_training_++;
        } else {
            pair<int, float> pred = mf->predict_class_confident(sample);
            if (pred.second < hp.active_confidence_value_) {
                mf->update(sample);
                pResult_ -> samples_used_for_training_++;
            }
        }
        /* Show progress */
        ++show_progress;
    }
  } else if (hp.active_learning_ == 2) {

    /* Active learning with buffering samples to learn only samples that are very
     * uncertain (last x%)*/

    pair<Sample, float> i_active_sample;
    list<pair<Sample, float> > active_buffer;
    int count_buffer = 0;

    for (int long i_samp = 0; i_samp < number_training_samples; i_samp++) {
      Sample sample = dataset.get_next_sample();

      if (i_samp < hp.active_number_init_set_) {
        mf->update(sample);
        pResult_ -> samples_used_for_training_++;
      } else {

        pair<int, float> pred = mf->predict_class_confident(sample);
        i_active_sample.first = sample;
        i_active_sample.second = pred.second;
        /* Insert sample */
        insert_sort(active_buffer, i_active_sample);
        count_buffer++;
        
        if (count_buffer >= hp.active_batch_size_) {
          /* Go through active buffer and update "active_buffer" of most uncertain
           * samples */
          list<pair<Sample, float> >::iterator it = active_buffer.begin();
          for (int i_buf = 0; it != active_buffer.end(); it++) {
            mf->update((*it).first);
            pResult_ -> samples_used_for_training_++;
            if (i_buf == 0) 
                active_conf_values.push_back((*it).second);
            if (i_buf == hp.active_buffer_size_){
                active_conf_values.push_back((*it).second);
                break;
            }
            ++i_buf;
          }
          count_buffer = 0;
          active_buffer.clear();
        }
      }
      /* Show progress */
      ++show_progress;
    }


  } else {
    cout << "[Error: option for active learning is not available]" << endl;
    exit(EXIT_FAILURE);
  }

  /*---------------------------------------------------------------------*/
  cout << endl;
  cout << " ... finished training after: ";
  timeval endTime;
  gettimeofday(&endTime, NULL);
  float tmp_training_time = (endTime.tv_sec - startTime.tv_sec + 
          (endTime.tv_usec - startTime.tv_usec) / 1e6);
  cout << tmp_training_time << " seconds." << endl;

  pResult_ -> training_time_ += tmp_training_time;
}
/**
 * Function tests/evaluates a mondrian forest
 */ 
double Experimenter::test(MondrianForest* mf, DataSet& dataset,
    Hyperparameters& hp) {


  cout << endl;
  cout << "-----------------" << endl;
  cout << "Start testing ..." << endl;
  cout << "-----------------" << endl;
  cout << endl;

  /* Check if test file exists */
  if (dataset.num_samples_ < 1) {
      cout << "[ERROR] - There does not exist a test dataset." << endl;
      exit(EXIT_FAILURE);
  }

  /* Initialize stop time for training */
  timeval startTime;
  gettimeofday(&startTime, NULL);


  /* Initialize progress bar */
  unsigned int expected_count= dataset.num_samples_;
  /* Display training progress */
  boost::progress_display show_progress( expected_count );
  
  /*---------------------------------------------------------------------*/

  int pred_class = 0;  /* Predicted class */
  int conf_pos = 0;  /* Position of confidence value */

  /* Go through complete test set */
  for (unsigned int n_elem = 0; n_elem < dataset.num_samples_; n_elem++) { 

    /* Get next sample */
    Sample sample = dataset.get_next_sample();

    pred_class = 0;

    if (conf_value_) {
      /* 
       * Calculates a confidence value for each prediction and saves
       * it in some kind of bar representation for further visualization
       */
      pair<int, float> pred = mf->predict_class_confident(sample);
      pred_class = pred.first;
      conf_pos = int((pred.second * 100) / 5);

      if (conf_pos <= 20) {
        if (conf_pos == 20)
          conf_pos = 19;
        if (pred_class == sample.y) {
          pResult_ -> confidence_[conf_pos] += 1;
        } else {
          pResult_ -> confidence_false_[conf_pos] += 1;
        }
      } else {
        std::cout << "Warning: confidence value is wrong! " << conf_pos << 
          std::endl;
      }
    } else { 
      /* Prediction */
      pred_class = mf->predict_class(sample);
    }

    pResult_ -> result_prediction_.push_back(pred_class);

    /* Show progress */
    ++show_progress;
  }
  
  /*---------------------------------------------------------------------*/

  cout << endl;
  cout << " ... finished testing after: ";
  timeval endTime;
  gettimeofday(&endTime, NULL);
  float tmp_testing_time = (endTime.tv_sec - startTime.tv_sec + 
          (endTime.tv_usec - startTime.tv_usec) / 1e6);
  pResult_ -> testing_time_ += tmp_testing_time;
  cout << tmp_testing_time << " seconds." << endl;
  
  /* Evaluate test results */
  pResult_ -> accuracy_ = evaluate_results(dataset);

  return pResult_ -> accuracy_; 
}


/**
 * Evaluate test results
 */
double Experimenter::evaluate_results(DataSet& dataset_test) {

  dataset_test.reset_position();
  unsigned int same_elements = 0;
  for (unsigned int n_elem = 0; n_elem < pResult_ -> result_prediction_.size();
          n_elem++) {
      Sample sample = dataset_test.get_next_sample();
      if (pResult_ -> result_prediction_[n_elem] == sample.y) {
          same_elements++;
          pResult_ -> result_correct_prediction_.push_back(1);
      } else {
          pResult_ -> result_correct_prediction_.push_back(0);
      }
  }
  float accuracy = 0.0;
  if (same_elements != 0) {
      accuracy = (float) same_elements / pResult_ -> result_prediction_.size();
  } else {
      accuracy = 0.0;
  }
  return accuracy;
}
/**
 * Return training time
 */
double Experimenter::get_training_time() {
  return pResult_ -> training_time_;
}
/**
 * Return testing time
 */
double Experimenter::get_testing_time() {
  return pResult_ -> testing_time_;
}
/**
 * Return accuracy value
 */
double Experimenter::get_accuracy() {
  return pResult_ -> accuracy_;
}

/** 
 * Return detailed result
 */
Result Experimenter::get_detailed_result() {
  return (*pResult_);
}
