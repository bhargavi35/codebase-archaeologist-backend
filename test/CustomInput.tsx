import React from 'react';
export const CustomInput = ({ error }) => (
  <div>
    <input className="border p-2" />
    {error && <span className="text-red-500">{error}</span>}
  </div>
);