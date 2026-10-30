import{
  Ht as e,xn as t
}
from"./D-o1eNUi.js";
import{
  Cn as n,St as r,Wt as i,bt as a,y as o
}
from"./ja3FViFE.js";
import"./BdjTWeIk.js";
import{
  t as s
}
from"./DHwBUaWl.js";
import{
  t as c
}
from"./CiTTpuZf.js";
import{
  t as l
}
from"./DfRCsiE8.js";
import{
  r as u
}
from"./BO-EuQhV.js";
var d=new o,f=new n,p=(e,t,i)=>{
  let a=e.ref,o=new n;
  a instanceof r&&a.geometry?(a.geometry.computeBoundingBox(),a.geometry.boundingBox&&(a.geometry.boundingBox.getCenter(o),d.copy(a.geometry.boundingBox),d.applyMatrix4(a.matrixWorld),d.getCenter(f))):a.getWorldPosition(f);
  let s=a.position.clone(),c=a.quaternion.clone(),l=f.clone().sub(t),u=l.length(),p=new n(1,(Math.random()-.5)*.6,(Math.random()-.5)*.6).normalize(),m=(.5+Math.random())*(Math.random()>.5?1:-1);
  return u<1e-4&&l.set((Math.random()-.5)*.01,(Math.random()-.5)*.01,(Math.random()-.5)*.01),l.normalize(),{
    accumulatedRotation:0,distanceWeight:i>0?u/i:0,explodeDirection:l.clone(),explodeDistance:u,geometryCenterOffset:o.clone(),originalLocalPosition:s.clone(),originalRotation:c.clone(),rotationAxis:p.clone(),rotationSpeedFactor:m
  }
  
}
,m=new n,h=new n,g=new n,_=new i,v=new a,y={
  Collapsing:`collapsing`,Expanding:`expanding`,Idle:`idle`,Rotating:`rotating`
}
,b=(e=y.Idle,t=0,n=new Map)=>({
  progress:t,rotationSnapshot:n,startProgress:t,startTime:e===y.Idle?0:performance.now(),state:e
}
),x=e=>1+2.2*(e-1)**3+1.2*(e-1)**2,S=e=>e<.5?4*e*e*e:1-(-2*e+2)**3/2;
function C(t){
  let i=null;
  e(()=>[t.enabled(),t.collapsing()],()=>i?.sync(),{
    flush:`sync`
  }
  );
  function a(e,a={
    enabled:!0
  }
  ){
    let s=a,c=t.entities().filter(t=>e.content.meshes.includes(t.ref)),l=new Map,u=null,d=!1,f=!1,C=()=>{
      if(u===null)return;
      let e=u;
      u=null,t.selectionEnabled()||t.setSelectionEnabled(e)
    }
    ,w={
      value:!1
    }
    ,T=new n,E=0,D=b(),O=e=>{
      let t=new o;
      e.forEach(e=>{
        if(e.ref instanceof r&&e.ref.geometry&&(e.ref.geometry.computeBoundingBox(),e.ref.geometry.boundingBox)){
          let n=e.ref.geometry.boundingBox.clone();
          n.applyMatrix4(e.ref.matrixWorld),t.union(n)
        }
        
      }
      );
      let i=new n;
      t.getCenter(i);
      let a=new n,s=t.getSize(a).length()*.03,c=0,l=0;
      return e.forEach(e=>{
        if(e.ref instanceof r){
          let t=new n,r=!1;
          if(e.ref.geometry?.boundingBox){
            let n=e.ref.geometry.boundingBox.clone();
            n.applyMatrix4(e.ref.matrixWorld),r=n.getSize(a).length()<s,n.getCenter(t)
          }
          else e.ref.getWorldPosition(t);
          let o=t.distanceTo(i);
          o>l&&(l=o),!r&&o>c&&(c=o)
        }
        
      }
      ),c===0&&(c=l),{
        center:i,maxDistance:c
      }
      
    }
    ,k=(e,t,n)=>{
      e.forEach(e=>{
        if(e.ref instanceof r&&!l.has(e)){
          let r=p(e,t,n);
          l.set(e,r)
        }
        
      }
      )
    }
    ,A=()=>{
      if(!c||c.length===0){
        w.value=!1,t.setInitialized(!1);
        return
      }
      let{
        center:e,maxDistance:n
      }
      =O(c);
      T.copy(e),E=n,k(c,T,E),w.value=!0,t.setInitialized(!0)
    }
    ,j=()=>{
      c&&c.forEach(e=>{
        let t=l.get(e);
        t&&(e.ref.position.copy(t.originalLocalPosition),e.ref.quaternion.copy(t.originalRotation),t.accumulatedRotation=0)
      }
      )
    }
    ,M=()=>{
      C(),j(),D=b()
    }
    ,N=(e,t,n)=>b(t,e.progress,n??e.rotationSnapshot),P=(e,n)=>{
      if(e.state===y.Idle||e.state===y.Rotating)return e;
      let r=n-e.startTime,i=Math.min(r/800,1),a=+(e.state===y.Expanding),o=e.startProgress+(a-e.startProgress)*i;
      if(i>=1){
        if(e.state===y.Expanding)return N({
          ...e,progress:1
        }
        ,y.Rotating);
        if(e.state===y.Collapsing)return j(),t.disable(),b()
      }
      return{
        ...e,progress:o
      }
      
    }
    ,F=(e,t)=>{
      let n=(t-(1-e)*.25*.4)/.6;
      return Math.max(0,Math.min(1,n))
    }
    ,I=(e,t,n)=>e<1?0:.02*t.rotationSpeedFactor*n,L=(e,t,n)=>{
      let r=F(e.distanceWeight,t),i=x(r),a=I(r,e,n);
      return e.accumulatedRotation+=a,{
        easedProgress:i,tLocal:r
      }
      
    }
    ,R=e=>{
      let t=.35+.65*Math.min(1,e.distanceWeight);
      return 1*E*t
    }
    ,z=(e,t,n,r)=>(r.copy(t.explodeDirection).multiplyScalar(n*R(t)),e.parent&&(e.parent.updateWorldMatrix(!0,!1),v.setFromMatrix4(e.parent.matrixWorld).invert(),r.applyMatrix3(v)),r),B=(e,t,n)=>{
      let r=z(e,t,n,m);
      return{
        pos:h.copy(t.originalLocalPosition).add(r),rot:t.originalRotation
      }
      
    }
    ,V=(e,t)=>{
      _.setFromAxisAngle(t.rotationAxis,t.accumulatedRotation),_.premultiply(t.originalRotation);
      let n=t.geometryCenterOffset,r=m.copy(n).multiply(e.scale).applyQuaternion(t.originalRotation),i=h.copy(n).multiply(e.scale).applyQuaternion(_),a=g.copy(r).sub(i),o=m.copy(t.originalLocalPosition).add(a),s=z(e,t,1,h);
      return o.add(s),{
        pos:o,rot:_
      }
      
    }
    ,H=(e,t,n)=>{
      _.setFromAxisAngle(t.rotationAxis,t.accumulatedRotation),_.premultiply(t.originalRotation);
      let r=t.geometryCenterOffset,i=m.copy(r).multiply(e.scale).applyQuaternion(t.originalRotation),a=h.copy(r).multiply(e.scale).applyQuaternion(_),o=g.copy(i).sub(a),s=z(e,t,1-n,m);
      return{
        pos:h.copy(t.originalLocalPosition).add(o).add(s),rot:_
      }
      
    }
    ,U=(e,t)=>{
      e.position.copy(t.pos),e.quaternion.copy(t.rot)
    }
    ,W=(e,t,n,r)=>{
      let i=n>=1?V(e.ref,t):B(e.ref,t,r);
      U(e.ref,i)
    }
    ,G=(e,t,n)=>{
      let r=t.state===y.Collapsing,i=S(r?Math.min((performance.now()-t.startTime)/800,1):0);
      e.forEach(e=>{
        let a=l.get(e);
        if(!a)return;
        if(r){
          a.accumulatedRotation=(t.rotationSnapshot.get(e)??a.accumulatedRotation)*(1-i);
          let n=H(e.ref,a,i);
          U(e.ref,n);
          return
        }
        let{
          easedProgress:o,tLocal:s
        }
        =L(a,t.progress,n);
        W(e,a,s,o)
      }
      )
    }
    ,K=()=>{
      l.clear(),e.content.root.updateWorldMatrix(!0,!0),A(),u=t.selectionEnabled(),t.setSelectionEnabled(!1),D=N(b(),y.Expanding),d=!0
    }
    ,q=()=>e.scope.isCurrent()&&t.model.captureContent()===e.contentId,J=()=>{
      if(f&&q()){
        if(!s.enabled||!t.enabled()){
          d&&M(),d=!1;
          return
        }
        if(d||K(),t.collapsing()&&D.state!==y.Collapsing){
          let e=new Map;
          for(let[t,n]of l)e.set(t,n.accumulatedRotation);
          D=N(D,y.Collapsing,e)
        }
        
      }
      
    }
    ;
    e.platform.onFrame(({
      delta:t
    }
    )=>{
      e.scope.isActive()&&q()&&d&&w.value&&(D=P(D,performance.now()),D.state!==y.Idle&&G(c,D,t))
    }
    );
    let Y=()=>{
      M(),d=!1,t.setInitialized(!1)
    }
    ,X={
      reset:Y,sync:J
    }
    ;
    e.scope.signal.addEventListener(`abort`,Y,{
      once:!0
    }
    );
    let Z=e.scope.defer(()=>{
      i===X&&(i=null),f=!1,Y(),e.scope.signal.removeEventListener(`abort`,Y),l.clear()
    }
    );
    return{
      activate:()=>{
        f=!0,i=X,J()
      }
      ,deactivate:()=>{
        Z()
      }
      ,update:e=>{
        s=e,J()
      }
      
    }
    
  }
  return{
    forceReset:()=>i?.reset(),prepare:a
  }
  
}
var w=s(`explode`,({
  enable:e,onDispose:n,onInit:r
}
)=>{
  let i=c(),a=l(),o=t(!1),s=t(!1),d=C({
    collapsing:()=>s.value,disable:()=>{
      e.value=!1,s.value=!1
    }
    ,enabled:()=>e.value,entities:()=>i.entityMesh,model:i.controller,selectionEnabled:()=>a.enable,setInitialized:e=>{
      o.value=e
    }
    ,setSelectionEnabled:e=>{
      a.enable=e
    }
    
  }
  ),f=()=>{
    d.forceReset(),e.value=!1,s.value=!1
  }
  ;
  return r(()=>{
    e.value=!1
  }
  ),n(i.controller.onContentRelease(()=>{
    f()
  }
  )),u(`effect:explode`,t=>{
    t?(s.value=!1,e.value=!0):s.value=!0
  }
  ),{
    enable:e,forceReset:f,isInitialized:o,prepareBehavior:d.prepare
  }
  
}
);
export{
  w as t
}
;
