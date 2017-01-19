#include <iostream>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <algorithm>
#include <memory>
#include "deepForeach.hpp"
#include "isContainer.hpp"
#include <cstdlib>
#include "Supercell.hpp"
#include "Frame.hpp"
#include "Particle.hpp"
#include "DeepContainer.hpp"
template<
    typename TElement>
struct DeepIterator;


/**
 * @brief Der DeepContainer ist ein virtueller Container, welcher zusätzliche 
 * Informationen bereithält. Beispielsweiße benötigt ein Frame die Information,
 * wie viele Particle in diesem Frame drin sind. 
 * In diesem Beispiel wollen wir einen DeepContainer schreiben, welcher nur das 
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
 * 1. Ich überlege mir, wie ich den DeepContainer auf den std::vector abbilde
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



int main(int argc, char **argv) {
    std::vector<int> vec1d = {0, 1, 2, 3, 4, 5};
   // DeepContainer<std::vector<int>, int> deepCon1d(vec1d);
    typedef Particle<int, 2> particle_type;
    typedef Frame<particle_type, 10> frame_type;
    typedef SuperCell<frame_type> supercell_type;
    deepForeach(vec1d, [](int& x){std::cout << x << " ";});

    std::cout << std::endl;
    
    SuperCell< frame_type > cell(5, 2);
 
    DeepContainer<frame_type, particle_type > con(*cell.lastFrame, 2);
    
    std::for_each(con.begin(), con.end(), [](const particle_type& par){std::cout << par;});

    std::cout << std::endl <<"output of frames in supercell" << std::endl;
    
    DeepContainer<supercell_type, frame_type > con2(cell);
    
    std::for_each(con2.begin(), con2.end(), [](const frame_type& par){std::cout << par;});
    
    return EXIT_SUCCESS;
    
}
