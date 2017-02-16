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
#include "DeepView.hpp"
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



int main(int argc, char **argv) {

   // <std::vector<int>, int> deepCon1d(vec1d);
    typedef hzdr::Particle<int, 2> Particle;
    typedef hzdr::Frame<Particle, 10> Frame;
    typedef hzdr::SuperCell<Frame> Supercell;



    Supercell cell(5, 2);
    
    std::vector<Supercell> supercellContainer{ Supercell(5,1), Supercell(7,4), Supercell(3,4)};
    
    std::cout << cell << std::endl;
 
    hzdr::DeepView<Frame, Particle, hzdr::Direction::Forward,hzdr::Collectivity::NonCollectiv, 3> con(*cell.firstFrame, 2);
    
    for(auto it = con.begin(); it != con.end(); ++it)
    {
        std::cout << *it;
    }

 //   hzdr::Frame<hzdr::Particle<int, 2u>, 10u> t;
 //   hzdr::Accessor<hzdr::Frame<hzdr::Particle<int, 2u>, 10u> > test(2);
  // std::cout << test;
  //  con.begin();
     hzdr::deepForeach(con, [](const Particle& par){std::cout << par;});

    std::cout << std::endl <<"output of frames in supercell" << std::endl;
    
    hzdr::DeepView<Supercell, Frame, hzdr::Direction::Forward,hzdr::Collectivity::NonCollectiv, 1 > con2(cell);
    

    
     hzdr::deepForeach(con2, [](const Frame& par){std::cout << par << std::endl;});
    
    
    /**
     * @todo Zusammensetzen des Iterators über alle Particle in Supercellen
    */
    typedef hzdr::DeepView<Frame, Particle, hzdr::Direction::Forward,hzdr::Collectivity::NonCollectiv, 1> ParticleContainer;
    typedef hzdr::DeepView<Supercell, Frame,  hzdr::Direction::Forward,hzdr::Collectivity::NonCollectiv, 1> FrameInSuperCellContainer;
    
    
    typedef hzdr::DeepView<Supercell, 
                                ParticleContainer,
                                hzdr::Direction::Forward, 
                                hzdr::Collectivity::NonCollectiv,
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
    
    for(auto it=con3.begin(); it!=con3.end(); ++it)
    {
        
    }
    /*
    for(auto it=con3.begin(); it!=con3.end(); ++it)
    {
        
    }
    */
    
  //  hzdr::deepForeach(con3, [](const Particle& par){std::cout << par << " a";});

    
   // typeid.name();
   // con3.end();
   /* 
    auto test = con3.begin();
  //  (*test).begin();
    //*/
   // std::cout << (con3.begin());
    
    return EXIT_SUCCESS;
    
}
