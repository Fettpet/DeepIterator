#include <iostream>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <algorithm>
#include <typeinfo>
#include <memory>
#include "deepForeach.hpp"
#include "Traits/isContainer.hpp"
#include <cstdlib>
#include "PIC/Supercell.hpp"
#include "PIC/Frame.hpp"
#include "PIC/Particle.hpp"
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

   // DeepContainer<std::vector<int>, int> deepCon1d(vec1d);
    typedef Data::Particle<int, 2> particle_type;
    typedef Data::Frame<particle_type, 10> frame_type;
    typedef Data::SuperCell<frame_type> supercell_type;



    Data::SuperCell< frame_type > cell(5, 2);
    
    std::cout << cell << std::endl;
 
    Data::DeepContainer<frame_type, particle_type, Data::Direction::Forward, 3> con(*cell.firstFrame, 2);

 //   Data::Frame<Data::Particle<int, 2u>, 10u> t;
 //   Data::Accessor<Data::Frame<Data::Particle<int, 2u>, 10u> > test(2);
  // std::cout << test;
  //  con.begin();
     Data::deepForeach(con, [](const particle_type& par){std::cout << par;});

    std::cout << std::endl <<"output of frames in supercell" << std::endl;
    
    Data::DeepContainer<supercell_type, frame_type, Data::Direction::Forward, 1 > con2(cell);
    

    
     Data::deepForeach(con2, [](const frame_type& par){std::cout << par << std::endl;});
    
    
    /**
     * @todo Zusammensetzen des Iterators über alle Particle in Supercellen
    */
    typedef Data::DeepContainer<frame_type, particle_type, Data::Direction::Forward, 1> ParticleContainer;
    typedef Data::DeepContainer<supercell_type, frame_type,  Data::Direction::Forward, 1> FrameInSuperCellContainer;
    
    
    typedef Data::DeepContainer<supercell_type, 
                                ParticleContainer,
                                Data::Direction::Forward, 
                                1
                        > ParticleInSuperCellContainer;
    
    /**
     * Ich gebe eine Superzelle rein und erhalte einen Particle zurück
     * Superzelle -> frame -> particle
     * 
     * Durch den Container wird die Reihenfolge
     * Container < Superzelle, Frame > -> Container < Frame, Particle >
     *
     * Also wäre die sinnvollste Reihenfolge
     * Container < Superzelle, Container< Frame, Particle > >
     * 
     */ 
   
    
    ParticleInSuperCellContainer con3(cell);
    
    
    Data::deepForeach(con3, [](const particle_type& par){std::cout << par << " a";});

    
   // typeid.name();
   // con3.end();
   /* 
    auto test = con3.begin();
  //  (*test).begin();
    //*/
   // std::cout << (con3.begin());
    
    return EXIT_SUCCESS;
    
}
