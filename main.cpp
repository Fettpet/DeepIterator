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


    typedef hzdr::Particle<int, 2> Particle;
    typedef hzdr::Frame<Particle, 10> Frame;
    typedef hzdr::SuperCell<Frame> Supercell;

    Supercell cell(5, 2);
    std::cout << cell << std::endl; 
    // All Particle within a Supercell

    typedef hzdr::View<Frame, 
                       hzdr::Direction::Forward, 
                       hzdr::Collectivity::None, 
                       hzdr::runtime::TupleFull> ParticleInFrame;
    

    typedef hzdr::View<Supercell,
                       hzdr::Direction::Forward, 
                       hzdr::Collectivity::None, 
                       hzdr::runtime::TupleFull,
                       ParticleInFrame> ParticleInSuperCell;
                       
    hzdr::runtime::TupleFull runtimeFrame(0, 1, 1);
    hzdr::runtime::TupleFull runtimeSupercell(1, 1, 1);

   ParticleInSuperCell test(cell, runtimeSupercell, ParticleInFrame(cell.firstFrame, runtimeSupercell));
   ParticleInFrame t(cell.firstFrame, runtimeFrame);
   t.begin();
   t.end();
   for(auto it=test.begin(); it!=test.end(); ++it)
   {
        std::cout << **it;
   }
/*
   for(auto it=test.begin(); it!=test.end(); ++it)
   {
        std::cout << **it;
   }
    /*
    for(auto it=test.begin(); it!=test.end(); ++it)
    {
        if(*it)
        {
            std::cout << **it << std::endl;
        }
    }
  //  ParticleInFrame(2);
   //  test(cell, 2, ParticleInFrame(nullptr, 2)); 
    

//    std::cout << std::endl << counter << std::endl;
    /**
     * Wie stelle ich mir den Aufruf des Verschachtelten Iterators vor?
     * 1. Beispiel: Alle Particles in Superzellen:
     * 
     * View < Supercell, jumpsize, Direction, View< Frame, jumpsize, Direction, Collectivity> >
     */
    
    

   // Particle test(5,4);
  /*  
    hzdr::View<Particle, hzdr::Direction::Forward, hzdr::Collectivity::NonCollectiv, 1> con(&test, static_cast<uint_fast32_t>(2));


    for(auto it=con.begin(); it!=con.end(); ++it)
    {
        auto wrap = *it;
        if(wrap)
        {
            std::cout << *wrap << std::endl;
        }
        
    }
    
    
 //   hzdr::Frame<hzdr::Particle<int, 2u>, 10u> t;
 //   hzdr::Accessor<hzdr::Frame<hzdr::Particle<int, 2u>, 10u> > test(2);
  // std::cout << test;
  //  con.begin();
  
  /*

    std::cout << std::endl <<"output of frames in supercell" << std::endl;
    
    hzdr::View<Supercell, Frame, hzdr::Direction::Forward,hzdr::Collectivity::NonCollectiv, 1 > con2(cell);
    
    for(auto it = con2.begin(); it != con2.end(); ++it)
    {
        auto wrap =*it;
        if(wrap)
        {
            std::cout << *(wrap) << std::endl;
        }
    }
    
*/
    /*
    
    typedef hzdr::View<Frame, Particle, hzdr::Direction::Forward,hzdr::Collectivity::NonCollectiv, 1> ParticleInFrameContainer;
    typedef hzdr::View<Supercell, Frame,  hzdr::Direction::Forward,hzdr::Collectivity::NonCollectiv, 1> FrameInSuperCellContainer;
    
    
    typedef hzdr::View<Supercell, 
                                ParticleInFrameContainer,
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
   /*
    
    ParticleInSuperCellContainer con3(cell);
    
    for(auto it = con3.begin(); it != con3.end(); ++it)
    {
        auto wrap =*it;
        if(wrap)
        {
            std::cout << *wrap << std::endl;
        }
    }
*/
    return EXIT_SUCCESS;
    
}
