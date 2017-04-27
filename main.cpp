#include <iostream>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <algorithm>
#include <typeinfo>
#include <memory>
#include <cstdlib>
#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
#include "View.hpp"
#include "Traits/HasOffset.hpp"
#include <omp.h>
#include "Iterator/RuntimeTuple.hpp"
template<
    typename TElement>
struct DeepIterator;


/**
 * @brief Der  ist ein virtueller Container, welcher zusätzliche 
 * Informationen bereithält. Beispielsweiße benötigt ein Frame die Information,
 * wie viele Particle in diesem Frame drin sind. 
 * In diesem Beispiel wollen wir einen  schreiben, welcher nur das 
 * n-te Element betrachtet. Als erstes betrachten wir hierfür den .
 * 
 * Zu betrachtende Beispiele
 * 1. 1d Fall (done
 * 2. 2d Fall
 * 3. nd Fall
 * 4. 1d Fall nur jedes zweites Element
 * 
 * 
 * 
 * 1. Fall
 * Arbeitsreihenfolge:
 * 1. Ich überlege mir, wie ich den  auf den std::vector abbilde
 * 
 * Bedingungen:
 * Damit der Deepcontainer mit DeepForeach arbeitet benötigt er
 * 1. begin() und end()
 * 2. einen definierten iterator datentype (innerhalb der Klasse)
 * 
 * 2. Fall
 * 
 * @tparam TContainer ist der Container Der Daten
 * @tparam TElement Der Rückgabetyp des Iterator. in unserem ersten Beispiel ist
 * es ein int
 */

struct testTrue 
{
    float offset;
};



int main(int argc, char **argv) {


    typedef hzdr::Particle<int_fast32_t, 2> Particle;
    typedef hzdr::Frame<Particle, 10> Frame;
    typedef hzdr::SuperCell<Frame> Supercell;

    Supercell cell(5, 2);
    std::cout << cell << std::endl; 
    // All Particle within a Supercell

    


                       
    
    
    hzdr::runtime::TupleFull runtimeSupercell(2, 0, 0);
    hzdr::runtime::TupleOpenMP runtimeFrame(2);

   
   
    typedef hzdr::View<Supercell,
                       hzdr::Direction::Forward, 
                       hzdr::Collectivity::None, 
                       hzdr::runtime::TupleFull> supercellView;
                       
    supercellView view(cell, runtimeSupercell);
    std::cout << cell << std::endl;
    for(auto it=view.begin(); it!=view.end(); ++it)
    {
        if(*it)
            std::cout << "Frame: " << **it << std::endl;
    }

    return EXIT_SUCCESS;
    
}
