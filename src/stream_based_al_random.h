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

#ifndef STREAM_BASED_AL__RANDOM_H_
#define STREAM_BASED_AL__RANDOM_H_

/*
 * Used to generate random numbers
 */
#include <boost/random/beta_distribution.hpp>
#include <boost/random/exponential_distribution.hpp> 
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>  /**< Random generator mt19937 */
#include <boost/generator_iterator.hpp>
#include <boost/random/uniform_real.hpp>
#include <sys/time.h>

#include "stream_based_al_utilities.h"

using namespace std;

/*---------------------------------------------------------------------------*/
typedef boost::mt11213b base_generator_type;

/*---------------------------------------------------------------------------*/
class RandomGenerator {

    public:

        RandomGenerator();

        float rand_uniform_distribution();
        /**
         * Generate value that is uniform distributed between min and max
         */
        float rand_uniform_distribution(float min_value, float max_value);
        /**
         * Generate value that is uniform distributed between min and max
         *
         * @param equal_values  : return true if min_value == max_value
         */
        float rand_uniform_distribution(float min_value, float max_value,
                bool& equal_values);
        float rand_exp_distribution(float lambda);
        float rand_beta_distribution(float alpha, float beta);

    private:
        static bool seed_flag_;
        static base_generator_type generator;

        boost::uniform_real<float> uni_dist;
        boost::exponential_distribution<float> exp_dist;
        boost::variate_generator<base_generator_type&,
            boost::uniform_real<float> > uni_gen;

};

#endif /* STREAM_BASED_AL__RANDOM_H_ */
/*---------------------------------------------------------------------------*/

inline double init_seed() {
    ifstream devFile("/dev/urandom", ios::binary);
    unsigned int outInt = 0;
    char tempChar[sizeof(outInt)];

    devFile.read(tempChar, sizeof(outInt));
    outInt = atoi(tempChar);

    devFile.close();

    struct timeval TV;
    gettimeofday(&TV, NULL);
    double seed = TV.tv_sec * TV.tv_usec + getpid() + outInt;
    return seed;
}
